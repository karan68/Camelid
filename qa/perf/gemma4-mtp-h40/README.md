# Windows H40 — matched harness for the Gemma 4 26B-A4B MTP lane

The Windows/CUDA counterpart of the Mac `hybrid-hot40-experiment-runner`. It exists so
that a throughput number from this box can be put next to one from the M4 without either
side having to be taken on trust.

```powershell
# once, on the plain lane: establish what this lane's target actually emits
.\run-h40.ps1 -Arm plain -Establish

# thereafter, every arm is gated against that
.\run-h40.ps1 -Arm mtp-k8
.\run-h40.ps1 -Arm mtp-k1        # structural gate: must equal plain, exactly
```

## What is matched, and what is not

Matched: the fixture (`request-48-plain.json` is byte-identical to the Mac's,
`sha256 3a5562e8…`), chat templating through the serve lane's own renderer, greedy
decoding, 48 tokens, a token-id gate, an idle-state admission check before the run, and
host facts recorded on both sides of every measurement.

Not matched, and not faked: the Mac's `vm_stat` pressure levels, wired-memory ceiling,
`F_RDADVISE` policy and `lsof` port observation have no Windows equivalent. Where a
weaker Windows stand-in exists it is recorded and labelled as one — the hard-fault page
counters in `host_delta` are system-wide and include ordinary mapped-file reads, so they
are evidence, not the swap gate the Mac harness enforces.

## The two expectation files

| file | role |
|---|---|
| `expected-48-token-ids.mac.json` | The frozen Metal reference, 48 ids. **Always compared, never the gate.** |
| `expected-48-token-ids.windows.json` | This lane's own plain-decode output, written by `-Establish`. **This is the gate.** |

The split is the point. MTP is lossless *against the target it actually runs*: every
emitted token is that target's own argmax, so a speculative arm must reproduce the plain
lane exactly, and any divergence there is a verifier or KV defect. Whether the Windows
target agrees with the Mac target is a **separate** question about cross-platform
runtime parity — one the perf handoff already answers partly (Camelid CPU and Camelid
CUDA agree with each other while both differ from llama.cpp on this row). Gating a
Windows speed run on Mac ids would conflate the two and stall on the wrong bug.

Both comparisons appear in every `verdict.json`, so neither can be quietly skipped.

## Arms

Arms live in `arms/*.json` and carry both the environment and the CLI shape, so an arm
is reproducible from one file. All arms share the same environment; only the drafter
differs, which is what keeps them comparable.

| arm | what it is |
|---|---|
| `plain` | Plain greedy decode. The reference every MTP arm has to beat. |
| `mtp-k8` | Speculative decode at K=8, the width the Metal campaign promoted. |
| `mtp-k1` | One draft per round. A correctness gate, not a speed arm — it exercises the whole speculative path with a single-position verify batch, so its ids must equal `plain`. |

## What a run produces

`runs/<stamp>-<arm>/` holds `stdout.txt`, `stderr.txt`, `receipt.json` (written by the
binary) and `verdict.json` (the receipt plus host state and both id comparisons).

The receipt splits the decode wall three ways — `prefill_ms`, `assistant_ms`,
`verify_ms` — because alpha alone cannot answer the only question that matters for this
lane. A round pays for itself iff assistant + verifier beats `1 + alpha` plain decode
steps; a receipt with alpha but no wall split cannot tell you whether a higher alpha
would have helped.

`rounds[]` records, per round, every draft beside the target's own argmax at the same
position, both read out of the *same* verify pass. That is what makes an alpha shortfall
attributable: if the drafts follow one token stream and the Windows target another, the
defect is target parity; if the drafts follow neither, it is assistant numerics.

## Reading the numbers

- **A single run is a receipt, not evidence.** Sequential GPU A/Bs on this laptop drift
  thermally; that once manufactured a phantom 1.8× win. Pair, alternate, and cool.
- **Record available RAM every time.** Numbers on this lane have ranged 4.3 → 23 tok/s
  on identical code depending on how much host RAM was free at load.
- **`decode_tokens_per_second` excludes prefill; `end_to_end_tokens_per_second` does
  not.** Say which one you are quoting.
- **A tok/s number without `exact_match: true` is not a result.**
