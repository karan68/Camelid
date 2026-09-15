# Model loading and memory admission

The default CPU weight-storage ceiling is the machine's detected physical RAM,
not a fixed number of GiB. `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES` remains
an explicit operator override. Missing hardware telemetry produces an actionable
error instead of inventing a capacity. Existing load-fit and runtime KV/scratch
checks continue to account for current memory pressure and the selected backend.
Readiness uses capacity rather than remaining free RAM so already-loaded weights
are not counted twice and a successful load does not disable Chat afterward.

`/v1/health` includes `generation_readiness_reason` for a blocked active model and
`model_load_progress` with filename, bytes read, and total bytes during a full file
check. Models and Chat display the reported progress. Load actions retain the
backend's readiness explanation instead of claiming a build cannot run a model.

The load-path SHA-256 cache is process-local and contains only hashes computed
from actual files by that process. On Unix, reuse requires an unchanged canonical
path, device, inode, length, modification time, and change time. A file changed
while hashing is rejected. Other platforms rehash when equivalent identity is
unavailable. No persistent cache can authorize an exact model capability;
verification continues to hash the file independently. First loads after launch
still read the whole artifact, with progress; unchanged repeat loads avoid that
read even after unloading or switching models.

## Validation

- Host-capacity/configuration, existing memory-estimation/admission, readiness,
  and concurrent health tests pass (18 focused Rust tests).
- Model-state, model-lane, capability-readiness, catalog activation, first-run
  activation, and rendered loading-status smoke checks pass.
- The frontend production build and the optimized v0.7.4 engine build pass.
- Render the UI regression with `cd frontend && node scripts/model-loading-status-smoke.mjs`.

First-use weight materialization runs on a blocking worker instead of the async
request executor. Large USB reads therefore leave the HTTP listener and health
polling schedulable. The model picker also waits for the engine's readiness
response; metadata alone is not treated as evidence that generation can run.

## Installed-build check on the affected M4 Mac

Both `Llama-3.2-3B-Instruct-Q4_K_M.gguf` and
`Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` returned “Hello!” twice with the default
memory policy and remained generation-ready after each reply. The test monitored
health throughout loading and generation; the largest observed response time was
5.1 ms.

The USB drive still determines cold I/O time: the 3B full-file check took 54.6 s;
the 8B check took 224.3 s, followed by 209.1 s of first-use weight loading. An
unchanged repeat metadata load after unloading took 0.71 s and 0.74 s respectively,
using the trusted process-local hash. These reload numbers do not include
rematerializing weights after an unload. The second chat on each resident model
reported a weight-cache hit and 0 ms weight loading.

The patched app is installed in `/Applications/Camelid Desktop.app`; quit and
reopen it to use the new engine. Installation keeps a timestamped backup under
`~/Library/Application Support/Camelid Backups/`. Raw checks and the installed
binary digest are in `qa/model-loading-memory-fix/results.json` and `install.json`.
