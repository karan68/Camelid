# Camelid frontend

This frontend targets Camelid. During the naming transition, the backend crate/binary and some API diagnostics may still use `backendinference`; commands below keep those current implementation names.

## Source of truth

This UI is the Camelid frontend. It was adapted from an existing local-model frontend implementation rather than rebuilt as a toy replacement, so future frontend work should preserve the mature app shell, views, components, styling, and UX structure while wiring behavior to Camelid.

The backend data hook is adapted for Camelid's current API surface:

- checks `GET /v1/health`
- lists `GET /v1/models`
- loads local GGUF paths through `POST /api/models/load`
- reads the support contract from `GET /api/capabilities`
- shows the support gate, current compatibility row, model-family/quantization evidence, and guarded API feature rows directly in chat, model setup, per-model/catalog cards, API, analytics, and system surfaces
- keeps the current runtime chat gate and `/api/capabilities` support gate visible in the page top bar outside the Chat/Models views, with a direct jump to the API contract before users interpret model-family or quant support
- keeps the API tab first-class in desktop/sidebar/mobile navigation, browser tab restore, and chat readiness prompts so the support contract is easy to find during readiness checks
- keeps API examples readiness-gated: `/api/capabilities` explains evidence boundaries, while `/v1/health` `loaded_now`/`generation_ready` plus `active_model_id` decide whether chat calls should run for the selected local GGUF
- normalizes loaded-model `general.file_type` values into GGUF quant labels (for example file type `7` → `Q8_0`) before comparing them to `/api/capabilities`, so loaded model cards get useful quant evidence without treating filenames as support claims
- keeps the shipped exact Llama 3.2 1B/3B Instruct Q8_0 and Llama 3 8B Instruct Q8_0 smoke rows visible as row-specific wins, while still requiring the loaded local GGUF to match its exact supported row before chat unlocks
- sends non-streaming chat requests to `POST /v1/chat/completions`
- blocks chat until `/v1/health` reports the selected `active_model_id` with `loaded_now: true` and `generation_ready: true` and `/api/capabilities` has an exact supported model/quant compatibility row; the exact Llama 3.2 1B/3B Instruct Q8_0 plus Llama 3 8B Instruct Q8_0 rows are supported only for their bounded local-chat smoke/parity envelopes

Server features Camelid does not expose yet are kept honest: catalog downloads, external-provider setup, planned/future/blocked quantization lanes, and unsupported or partial API parameters show disabled or typed-guardrail copy instead of pretending to work. The analytics view also treats conversation telemetry as usage only, not compatibility evidence. The UI mirrors the compatibility contract documented in `../COMPATIBILITY.md`; filenames, catalog metadata, saved browser paths, and prior usage are not treated as support evidence by themselves.

## Exact-row smoke wins shown in the UI

The frontend should make these shipped wins easy to see without turning them into broad Llama-family support:

- **Llama 3.2 1B Instruct Q8_0:** exact-row API/WebUI smoke plus compact and broader parity evidence are represented as a supported exact-row smoke lane.
- **Llama 3.2 3B Instruct Q8_0:** exact-row API/WebUI smoke, compact parity, broader three-prompt 50-token parity, and five-prompt API smoke are represented as a supported exact-row smoke lane.
- **Llama 3 8B Instruct Q8_0:** exact-row API/WebUI smoke, clean-main timing/RSS smoke, broader 50-token parity, the first bounded 512-context pack, compact chat-template-shapes pack, and measurement-only lazy-Q8 hot-path cost probe are represented as bounded exact-row wins without implying full-support performance portability.

All three rows still fail closed in the WebUI unless the active local GGUF matches the exact row and `/v1/health` reports `loaded_now=true` plus `generation_ready=true`. Do not infer support for neighboring sizes, base variants, other quantizations, arbitrary GGUF/Jinja templates, larger contexts, or performance portability from these cards.

## Run locally

Start Camelid first, usually on `127.0.0.1:8181`:

```bash
cargo run -- serve --addr 127.0.0.1:8181
```

Then run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:4175
```

## Configuration

The default API base is:

```text
http://127.0.0.1:8181
```

Override it at build/dev time with:

```bash
VITE_BACKENDINFERENCE_API_BASE=http://127.0.0.1:8181 npm run dev
```

You can also edit the API base in the UI sidebar while testing.

## Validation

Build the frontend:

```bash
cd frontend
npm run build
```

Smoke-test a running backend + frontend:

```bash
# terminal 1
cargo run -- serve --addr 127.0.0.1:8181

# terminal 2
cd frontend
npm run dev

# terminal 3
cd frontend
npm run smoke
```

For a self-contained local generation smoke test, use the tiny generated GGUF fixture:

```bash
cd frontend
npm run smoke:tiny
```

`smoke:tiny` creates a temporary tiny Camelid-compatible GGUF fixture, loads it through `POST /api/models/load`, verifies `generation_ready=true`, checks `/v1/models`, and confirms the WebUI chat guard stays blocked when that fixture does not have an exact supported `/api/capabilities` compatibility row. Real chat smoke only runs for models that are both `generation_ready=true` and support-contract matched.

To smoke-test a downloaded local GGUF without committing model files, pass its path explicitly:

```bash
cd frontend
npm run smoke -- --model ../models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --model-id tinyllama-q8
```

This verifies the frontend is reachable, loads the GGUF through the backend API, checks `/v1/health`, `/v1/models`, and the UI guardrails around `/api/capabilities`, and only sends a chat request when `generation_ready=true` **and** the active model has an exact supported compatibility row. The smoke output includes coarse timings for frontend reachability, model load, health/model listing, support-contract matching, and chat completion so real-model runs produce repeatable latency evidence. Add `--require-generation` when the model is expected to run end-to-end; otherwise the smoke exits successfully after confirming the UI/API guardrail state for metadata-only or unsupported-runtime models.

For the exact smoke-supported Llama rows, use the exact local path when a backend and frontend are running:

```bash
cd frontend
npm run smoke -- --model '$CAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf' --model-id llama-3.2-1b-instruct-q8 --require-generation --expect-compatibility-row llama32_1b_instruct_q8_0 --expect-compatibility-status supported_exact_row_smoke --expect-contract-supported true --expect-webui-chat enabled

npm run smoke -- --model '$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf' --model-id llama-3.2-3b-instruct-q8 --require-generation --expect-compatibility-row llama32_3b_instruct_q8_0 --expect-compatibility-status supported_exact_row_smoke --expect-contract-supported true --expect-webui-chat enabled

npm run smoke -- --model '$CAMELID_MODEL_DIR/Meta-Llama-3-8B-Instruct-Q8_0.gguf' --model-id llama-3-8b-instruct-q8 --require-generation --expect-compatibility-row llama3_8b_instruct_q8_0 --expect-compatibility-status supported_exact_row_smoke --expect-contract-supported true --expect-webui-chat enabled
```

These commands must still fail closed if the loaded model is the wrong row, lacks Q8_0 metadata, is not `loaded_now=true` + `generation_ready=true`, or is outside the exact supported `/api/capabilities` row. That is intentional: the UI supports only the exact 1B/3B/8B smoke rows without making a broad Llama-family claim. The latest public reopened-lane API + frontend smoke summary for all four exact rows is `../qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`; use it as frontend freshness evidence, not as broader/full-support evidence. The exact 8B broader three-prompt 50-token pack also passed at `../qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`, the first bounded 512-context pack passed at `../qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`, the compact chat-template-shapes pack passed at `../qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`, and the lazy-Q8 hot-path measurement is summarized at `../qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json`; these are bounded packs/measurements only and do not promote broader context, arbitrary-template behavior, production throughput, or performance portability.
