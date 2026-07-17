# Camelid Workspace — Qwen3 4B Q4_K_M real-model closure

This bundle closes the bounded Web Workspace vertical slice on the exact `Qwen3-4B-Q4_K_M.gguf` artifact. The model is pinned to SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` from `Qwen/Qwen3-4B-GGUF@a9a60d009fa7ff9606305047c2bf77ac25dbec49`.

## Passed scenarios

- read-only multi-step loop: `list_dir` → `read_file` → `search`;
- denied `write_file`: no file created and the workspace tree hash stayed unchanged;
- approved `write_file`: the approval UI displayed the exact target and complete proposed content, and the resulting file contained exactly `hello there`;
- an outside-root canary hash stayed unchanged;
- real browser approval and terminal-state screenshots were captured against the loaded model.

## Provenance

The run was captured from a dirty working tree based on `8c2a2b74e1db96ab67f593b3eb0c3628c8449ecf`. `manifest.json` records the implementation-file digest and the changed paths. This is evidence for that exact working tree, not a claim that the base commit alone contains the feature.

## Non-claims

No shell, network, GUI, subagent, unattended, neighboring-model, cross-platform, or throughput claim is made. The capability remains exact-row gated by committed `tool_capable` evidence.
