# Search by meaning on a 500-PDF library: index on the GPU, search in a quarter second (#801)

Search by meaning needs every chunk of the library embedded first. The Nomic
encoder ran only on the CPU, and on this machine it embedded 0.37 chunks a
second. The 500-PDF library from the #798 gate (550 PDFs, 114,083 chunks)
would have taken about 85 hours. Until a document is embedded, whole-library
search cannot see it at all, and a search by meaning cannot find it.

With this change, on an NVIDIA GPU, the same library is indexed in 24 minutes,
and a search by meaning takes about a quarter of a second.

## What changed

- **A CUDA lane for the encoder** (`src/embedding/cuda.rs`, CUDA builds only).
  The layer stack runs on the GPU, many chunks per forward. Tokenizing, the
  embedding rows, the first layer norm and pooling stay on the CPU and reuse
  the existing code. It is used for document indexing and for document-search
  queries; `/v1/embeddings` is unchanged.
  - The Q8 linears follow the CPU's operations exactly and return its bits.
  - Layer norm runs in f64 in the CPU's order, and RoPE uses the CPU's own
    cos/sin tables.
  - `exp` is taken in double precision and rounded once, to match the host's
    correctly rounded `expf`.
  - It falls back to the CPU:
    - when GPU acceleration is off;
    - when `CAMELID_DOCUMENT_GPU_ENCODER=0` is set;
    - when the lane cannot build;
    - for the rest of the process after any CUDA error.
  - A search never waits behind an indexing batch: if the GPU is busy, the
    query is embedded on the CPU.
- **Bigger batches on the GPU.** The indexer embeds 128 chunks per call once a
  batch has run on CUDA (16 on the CPU, as before). `GET
  /api/documents/index-status` now reports `semantic.lane` (`cpu` or `cuda`).
- **The library's vectors are kept in memory.** They are decoded once and
  reused until a chunk or vector changes. SQLite triggers bump a generation
  counter on every such change, so the cache is never used past one. The scan
  is parallel, and its scores are bit-identical to the old SQL path.
- **Index coverage is counted once per change.** The coverage a meaning search
  reports is now kept beside the cached vectors, instead of being counted by
  joining every chunk with its vector on every search.
- **Hybrid ties go to the keyword ranking.** When keyword and meaning fuse to
  exactly the same score, the passage the keyword ranking put higher wins.
  Before, the lower chunk id won, which is just whichever file was added
  first.

## How it was measured

One machine: 4 vCPUs, 15.6 GB RAM, an NVIDIA L4. The library is the #798
gate's: 500 arXiv PDFs and 50 planted PDFs, ingested by the #804 build,
114,083 chunks, no vectors. Every run started from a fresh copy of that
database. The server auto-loads Llama 3.2 3B (the first model in the models
directory) a few seconds after startup, so every run here shared the GPU with
a loaded chat model.

The questions are the gate's 50 planted ones. Each asks for an amount that one
planted PDF states. Another planted PDF holds a near-identical sentence (its
"decoy") about a neighbouring part, with a different amount.

### Indexing

| | Rate | The 114,083-chunk library |
|---|---|---|
| main (CPU), `data/cpu-indexing-rate-main.txt` | 224 chunks in 601 s = 0.37 chunks/s | about 85 hours (extrapolated) |
| this PR (CUDA), `data/index-cuda-final.txt` | 4,096 to 5,632 chunks a minute (78 a second overall) | 1,465 s, all 114,083 chunks |

Every minute of the CUDA run reported `lane: cuda`. The GPU itself does about
108 chunks a second (`data/cuda-parity-test.txt`). The rest of the server's
time goes to tokenizing, the embedding rows, pooling and SQLite.

On its own, in the test process, the encoder holds at most 858 MiB of GPU
memory (`data/cuda-encoder-gpu-memory.txt`). The 17.4 GB in the indexing log
is the auto-loaded chat model.

### The CUDA lane against the CPU

`embedding::tests::cuda_document_vectors_match_the_cpu` (ignored: needs the
encoder, a chunk file and a CUDA device), on 256 random chunks from the
library, gives `data/cuda-parity-test.txt`:

- One qkv projection: 0 of 182,016 values differ from the CPU.
- 48 whole document vectors: minimum cosine 0.9999587 to the CPU's, largest
  difference 1.2e-3. The test requires at least 0.9999. With CUDA's own
  `expf` it measured 0.99985, which is why `exp` is now taken in double
  precision.

### Search

`harness/retrieve_modes.py` runs every question four ways and records the rank
of the passage that holds the right amount, whether the decoy came first, and
the time.

| Search, 50 questions | main, library not indexed | this PR, indexed, before the tie-break | this PR, indexed (final) |
|---|---|---|---|
| Collection, default (hybrid once indexed) | first 50, top 5 50, decoy first 0 | first 36, top 5 50, decoy first 14 | first 44, top 5 50, decoy first 6 |
| Collection, meaning only | first 0, top 5 0 | first 32, top 5 49, decoy first 15 | first 32, top 5 49, decoy first 15 |
| Collection, keyword only | first 50, top 5 50 | first 50, top 5 50 | first 50, top 5 50 |
| Whole library | first 0, top 5 0 | first 36, top 5 50, decoy first 14 | first 44, top 5 50, decoy first 6 |

- **The default search on main** was keyword ranking: nothing was embedded
  yet. Whole-library search found nothing.
- **The 14 decoys first, before the tie-break,** were all 2nd-place facts, and
  in every one the fact was 1st by keyword. 8 were exact ties in the fused
  score (a tie-break changes nothing else), which had gone to the lower chunk
  id; the keyword tie-break puts the fact first in those 8.
- **In the 6 left,** meaning puts the fact 3rd or lower, so the decoy
  genuinely scores higher. Changing that would mean weighting keyword above
  meaning in general, which this change does not do.
- **The right passage is in the top 5** of every hybrid and whole-library
  search.

| Search time, p50 | main, not indexed | indexed, no cache (first build) | final |
|---|---|---|---|
| Collection, default | 3,867 ms | 1,562 ms | 494 ms |
| Collection, meaning only | 1,672 ms | 1,312 ms | 240 ms |
| Collection, keyword only | 537 ms | 256 ms | 257 ms |
| Whole library | 5,093 ms | 1,658 ms | 670 ms |

The main column ran while main's own CPU indexer was working through the
library, as it would be for days, so it includes that contention.
`data/search/time-*.txt` follow the steps, each on the indexed library after
the chat model had loaded:

| Step | Meaning only, whole collection | Meaning only, one document (mostly the query embedding) |
|---|---|---|
| vector cache, query on the CPU | 977 ms | 435 ms |
| + query on the GPU, parallel scan | 527 ms | 86 ms |
| + coverage counted once per change | 239 ms | 54 ms |
| final (+ tie-break) | 241 ms | 55 ms |

The second pass of the same searches (`data/search/indexed-final-second-pass.json`)
gives the same ranks.

### Chat

The gate's chat phase (`gate.py chat`, Llama 3.2 3B, `top_k` 4, the chat's
prompt on main) on the indexed library gives `data/chat-hybrid-final.json`.
It is compared with the keyword-only run in the #798 bundle
(`f2a-500-pdf-gate-20261008/data/chat-run1.json`), which was scored the same
way:

| Chat, 50 questions | keyword only (#798) | hybrid (this PR) |
|---|---|---|
| States the right amount | 50 | 48 |
| Also states the decoy's amount | 27 | 18 |
| Right, and a cited passage resolves to it | 30 | 32 |

### In the chat UI

`harness/live801.mjs` turns on **Whole library** in the real chat (this
branch's frontend, which this change does not touch) and asks two planted
questions in new chats.

| | main, library not indexed | this PR, indexed |
|---|---|---|
| The pill under the composer | Whole library, indexing 0/114083 | Whole library, 550 docs |
| Whole-library search | 0 passages | 4 passages, the right PDF first (both questions) |
| "Ivarsdal ... primary mirror" (335,300) | "I am unable to verify the cost..." | 335,300 euros (Citation 1), then also 937,700 (Citation 3) |
| "Wyncroft ... north clarifier" (487,200) | "I'm not able to find any information..." | 487,200 euros (Citation 1), then also 707,600 (Citation 2) |

The replies on this branch also list the decoy's amount as a second figure.
That is the prompt problem in #800, fixed by #805, which this branch (cut from
main) does not include. Screenshots: `screenshots/<run>-q<id>-1-asking.png`
and `-2-reply.png`.

## Not covered

- Metal: the lane is CUDA only. There is no Mac here, so Apple GPUs still
  index on the CPU.
- On the CPU the encoder is as before: 0.37 chunks a second on this machine.
- The vector cache holds every vector of the library in memory: 4 bytes per
  dimension, about 350 MB for these 114,083 chunks.
- The 6 decoys that still come first in a hybrid search are a ranking
  question, not a tie.

## Files

- `harness/`:
  - `retrieve_modes.py`, `final.sh`, `full-index.sh`, `index-rate.sh`,
    `latency.sh`, `serve.sh`;
  - `parity.sh` (runs the ignored test with a small environment, since the
    test build re-reads environment flags on every matmul);
  - `live801.mjs` and `live801.sh`;
  - `summarize.py` (prints the search tables from `data/search/`).
- `data/`: the runs above.
- `screenshots/`: the live run, 8 images.
