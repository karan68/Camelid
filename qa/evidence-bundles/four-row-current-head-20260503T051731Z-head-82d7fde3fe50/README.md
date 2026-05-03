# Full-support current-head execution bundle

Generated: 2026-05-03T05:17:31.572Z

Git head: `82d7fde3fe50b218f40ab18ef1123c81a985fc3f`
Origin/main: `e7837a49d3e1b2eb4f93ed6777b77200fad8307f`

This bundle is a durable execution scaffold for the four exact rows Tim cares about. It does **not** widen support by itself. Its job is to normalize the evidence shape so each row has the same folders, command files, model SHA capture, and carry-forward references before or during Ubuntu reruns.

Required tracks per row:
- compact parity
- broader parity
- chat-template shapes
- 512-context
- API/WebUI smoke
- perf/RSS/portability

Top-level commands:
- `commands/build-current-head.sh`
- `commands/capture-host-facts.sh`
- `commands/run-all-rows.sh`

Guardrails:
- Use the canonical Ubuntu validation host for promotion-grade Llama runtime evidence.
- Keep claims exact-row only unless docs, API, frontend, and artifacts all agree.
- Preserve known blockers durably instead of deleting them, especially the 8B 512-context timeout.

Carry-forward public references:
- `qa/evidence-bundles/four-row-public-20260503T024327Z`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md`
