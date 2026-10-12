# parity.sh <out file>: build the lib test binary, then run the CUDA parity test in a minimal env.
OUT=$1
export PATH=$HOME/.cargo/bin:$PATH
cd ~/camelid-801
BIN=$(CARGO_TARGET_DIR=$HOME/camelid-f2a-target cargo test --release --lib --no-run --message-format=json 2>/dev/null \
  | python3 -c 'import json,sys
for l in sys.stdin:
    m=json.loads(l)
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m["target"]["name"]=="camelid" and "lib" in m["target"]["kind"]:
        print(m["executable"])' | tail -1)
echo "binary $BIN" > "$OUT"
env -i HOME=$HOME PATH=/usr/bin:/bin CAMELID_NOMIC_GGUF=$HOME/Camelid/models/nomic-embed-text-v1.5.Q8_0.gguf \
  CAMELID_BENCH_CHUNKS=$HOME/r801/chunks256.jsonl "$BIN" --ignored --nocapture --exact \
  embedding::tests::cuda_document_vectors_match_the_cpu >> "$OUT" 2>&1
echo "EXIT $?" >> "$OUT"
