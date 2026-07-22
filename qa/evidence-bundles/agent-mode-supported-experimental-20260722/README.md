# Receipt bundle — agent mode, preview → Supported (experimental)

Live-lane evidence for promoting `camelid chat --agent` / `camelid agent exec`
from "(preview)" to **Supported (experimental)**, scoped to the receipted
battery on `tool_capable` rows. Model under test: the pinned CERT row
`qwen3_4b_instruct_q8_0` (`Qwen3-4B-Q8_0.gguf`, byte-identical to its
committed `qa/agent-eval` PASS receipt). Host: `host.json`.

## What "Supported (experimental)" claims — and does not

Claimed: the approval-gated agent loop on tool_capable rows — tool round-trips
(the `qa/agent-eval` battery), context compaction across a live boundary, the
opt-in stdio MCP lane end to end, and the headless `agent exec` tri-state
contract, each evidenced below or by the committed receipts cited.

Not claimed: rows without a current agent-eval PASS receipt; Windows/Linux
live-lane runs (CI-validated builds only); MCP servers beyond the stdio
transport; unattended (`--yolo`) operation beyond what the exec contract
states; any model quality bar beyond the battery.

## Files

- `compaction_live_*.txt` — G2 CERT: one `agent exec` run on the pinned row
  reads six 15.6 KB files sequentially; the transcript crosses the compaction
  budget SIX times (three clip-lane, three elide-into-Summary events visible
  in stderr), and the model still reads all six and answers correctly
  ("cobalt", the value in the last file). Exit 0.
- `mcp_live_*.txt` — G7 CERT: the same row, `--allow-mcp`, drives a namespaced
  stdio MCP tool (`mcp__parts__lookup_part`) through spawn → initialize →
  tools/list → gated call → fenced result → correct final answer
  ("42, bin B7"). Exit 0. `mcp_stub_server.py` is the exact server used.
- Promotion battery: `qa/agent-eval/Qwen3-4B-Q8_0-1784747759-PASS.json` —
  re-minted after the P0.5 prompt/fencing changes, on the full 3-case battery
  (the receipt it replaces covered 1 case). An earlier attempt on a contended
  box returned INCONCLUSIVE and changed no flag, as designed.

## Provenance

Transcripts are verbatim tool output with two mechanical substitutions:
`<home>` for the operator home directory and `<models-volume>` for the local
model-store mount (the privacy audit forbids durable volume paths). The runs used the debug binary at the head recorded in
`host.json`, with the default sandboxed shell, no `--allow-net`, and
`--allow-mcp`/`--yolo` only in the MCP CERT as noted in its stderr.
