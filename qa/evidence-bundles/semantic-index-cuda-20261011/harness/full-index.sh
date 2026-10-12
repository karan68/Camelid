#!/usr/bin/env bash
# full-index.sh: before/after for #801 on the 550-PDF library (114,083 chunks, no vectors yet).
set -u
R=~/r801
SRC=~/f2a-500/run-799
CORPUS=$SRC/corpus
count() { python3 -c "import sqlite3; c=sqlite3.connect('file:$1/documents_rag.sqlite3?mode=ro', uri=True); print(c.execute('select count(*) from document_chunk_vectors').fetchone()[0])"; }
vectors() { python3 -c "import sqlite3; c=sqlite3.connect('file:$1/documents_rag.sqlite3?mode=ro', uri=True); print(c.execute('select count(*) from document_chunk_vectors where vector is not null').fetchone()[0])"; }
chunks() { python3 -c "import sqlite3; c=sqlite3.connect('file:$1/documents_rag.sqlite3?mode=ro', uri=True); print(c.execute('select count(*) from document_chunks').fetchone()[0])"; }
wait_ready() { for _ in $(seq 1 120); do curl -s -m 2 -o /dev/null "127.0.0.1:$1/v1/health" && return; sleep 1; done; }

# 1. Before: main, library not indexed.
rm -rf $R/before && mkdir -p $R/before && cp $SRC/data/documents_rag.sqlite3 $R/before/
bash $R/serve.sh ~/camelid-803-base $R/before 18841 > $R/before.log 2>&1 || exit 1
wait_ready 18841
python3 $R/retrieve_modes.py 18841 $SRC/out $CORPUS $R/retrieve-before.json >> $R/before.log 2>&1
echo "vectors at the end of the before run: $(count $R/before)" >> $R/before.log
kill "$(cat $R/before/server.pid)"; sleep 3

# 2. Index with this branch.
rm -rf $R/after && mkdir -p $R/after && cp $SRC/data/documents_rag.sqlite3 $R/after/
bash $R/serve.sh ~/camelid-801-fix $R/after 18842 > $R/index.log 2>&1 || exit 1
wait_ready 18842
total=$(chunks $R/after)
start=$(date +%s)
curl -s 127.0.0.1:18842/api/documents/index-status | python3 -c "import json,sys; print('status', json.load(sys.stdin)['semantic'])" >> $R/index.log
prev=0
while :; do
  sleep 60
  now=$(count $R/after)
  lane=$(curl -s 127.0.0.1:18842/api/documents/index-status | python3 -c "import json,sys; s=json.load(sys.stdin)['semantic']; print(s.get('lane'), s.get('indexing'), s.get('error'))")
  gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')
  echo "t=$(( $(date +%s) - start ))s vectors $now/$total (+$((now - prev))) lane/indexing/error: $lane gpu: $gpu" >> $R/index.log
  prev=$now
  [ "$now" -ge "$total" ] && break
  [ $(( $(date +%s) - start )) -gt 10800 ] && { echo "gave up after 3 h" >> $R/index.log; break; }
done
echo "indexed $(count $R/after) of $total ($(vectors $R/after) with a vector) in $(( $(date +%s) - start )) s" >> $R/index.log

# 3. After: the same searches on the indexed library.
python3 $R/retrieve_modes.py 18842 $SRC/out $CORPUS $R/retrieve-after.json >> $R/index.log 2>&1
kill "$(cat $R/after/server.pid)"
echo DONE > $R/full-index.done
