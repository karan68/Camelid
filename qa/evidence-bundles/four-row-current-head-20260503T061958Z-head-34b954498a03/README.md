# Full-support current-head execution bundle

Generated: 2026-05-03T06:19:58.751Z

Git head: `34b954498a03eaca97cdb34e88419a7ddd54913e`
Origin/main: `ab3ee79fcd204717955c101569fc3a0871175be8`

This bundle is a durable execution scaffold for the four exact rows Tim cares about. It does **not** widen support by itself. Its job is to normalize the evidence shape so each row has the same folders, command files, model SHA capture, and carry-forward references before or during Ubuntu reruns.

Current status note: the 8B `context-512` blocker recorded in this scaffold has a later passing rerun at `../llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`. Keep the original blocker notes here as historical scaffold state, but use the later bundle for the current 8B single-pack context result.

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
- Preserve known blockers durably instead of deleting them; the original 8B 512-context timeout remains part of this scaffold history, while the later one-pack pass is cited separately.

Carry-forward public references:
- `qa/evidence-bundles/four-row-public-20260503T024327Z`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`
- `qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md`
