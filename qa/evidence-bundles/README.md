# Evidence bundles

This directory is for durable, reviewable evidence manifests and checksums.

Current public evidence map:
- `four-row-public-20260503T024327Z/` preserves the sanitized carry-forward smoke boundary.
- `four-row-perf-portability-public-20260503T025639Z/` preserves the compact perf/portability envelope.
- `four-row-current-head-20260503T061958Z-head-34b954498a03/` preserves the normalized current-head rerun scaffold and blocker notes.
- `four-row-api-only-20260504T230722Z-head-13a465608fbf/` is the reopened-lane API-only freshness slice with manifest and checksums.
- `four-row-api-webui-20260505T003100Z-head-b403884/` is the reopened-lane API + frontend smoke freshness slice for all four exact rows, with manifest and checksums.
- `full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/` is the current-head normalized TinyLlama/1B/3B API/WebUI smoke bundle from the reopened Ubuntu lane; it preserves manifest/checksum-verifiable evidence without broadening beyond exact-row smoke support.
- `llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/` closes only the bounded 8B broader three-prompt 50-token pack.
- `llama3-8b-context-512-20260504T234625Z-head-58acf592345c/` closes only the first bounded 8B 512-context pack.
- `llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/` closes only the bounded 8B compact chat-template-shapes pack.
- `llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/` is the clean-main exact 8B API/WebUI/RSS timing smoke for completion diagnostics; it does not widen support beyond the exact-row smoke envelope.
- `llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/` is the exact 8B retained-block lazy-Q8 hot-path cost probe; it is measurement evidence only, not a broader support/performance-portability promotion.
- `llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/` validates the reusable helper on clean public `main` and repeats the exact 8B retained-block Q8 measurements; it is still measurement evidence only.

Reproducibility helper:
- `node scripts/bench-q8-hotpath-bundle.mjs --model <model.gguf>` regenerates a sanitized retained-block Q8 hot-path bundle with per-tensor JSON, `manifest.json`, and `SHA256SUMS`. Use it for measurement staging only; pair results with production API/WebUI timing/RSS before making portability or throughput claims.

Rules:
- Commit only sanitized durable bundle content here.
- Keep raw/private staging copies out of git; they may contain private hostnames, home paths, or other operator-only details.
- Public bundles may point at `target/...` artifact roots, but they must not pretend those private raw trees are fetchable from GitHub.
- In committed manifests/checksums, prefer public-safe `qa/evidence-bundles/*-public-...` bundle paths over ignored raw bundle roots.
- Before citing or refreshing a durable bundle, run `node scripts/audit-evidence-bundle-privacy.mjs --root qa/evidence-bundles --out target/evidence-bundle-privacy-audit.json` and fix any findings.
