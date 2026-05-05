# Evidence bundles

This directory is for durable, reviewable evidence manifests and checksums.

Current public evidence map:
- `four-row-public-20260503T024327Z/` preserves the sanitized carry-forward smoke boundary.
- `four-row-perf-portability-public-20260503T025639Z/` preserves the compact perf/portability envelope.
- `four-row-current-head-20260503T061958Z-head-34b954498a03/` preserves the normalized current-head rerun scaffold and blocker notes.
- `four-row-api-only-20260504T230722Z-head-13a465608fbf/` is the reopened-lane API-only freshness slice with manifest and checksums.
- `four-row-api-webui-20260505T003100Z-head-b403884/` is the latest reopened-lane API + frontend smoke freshness slice for all four exact rows, with manifest and checksums.
- `llama3-8b-context-512-20260504T234625Z-head-58acf592345c/` closes only the first bounded 8B 512-context pack.

Rules:
- Commit only sanitized durable bundle content here.
- Keep raw/private staging copies out of git; they may contain private hostnames, home paths, or other operator-only details.
- Public bundles may point at `target/...` artifact roots, but they must not pretend those private raw trees are fetchable from GitHub.
- In committed manifests/checksums, prefer public-safe `qa/evidence-bundles/*-public-...` bundle paths over ignored raw bundle roots.
- Before citing or refreshing a durable bundle, run `node scripts/audit-evidence-bundle-privacy.mjs --root qa/evidence-bundles --out target/evidence-bundle-privacy-audit.json` and fix any findings.
