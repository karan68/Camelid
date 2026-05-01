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
- keeps API examples readiness-gated: `/api/capabilities` explains evidence boundaries, while `/v1/health` `loaded_now`/`generation_ready` decides whether chat calls should run
- normalizes loaded-model `general.file_type` values into GGUF quant labels (for example file type `7` → `Q8_0`) before comparing them to `/api/capabilities`, so loaded model cards get useful quant evidence without treating filenames as support claims
- keeps the exact Llama 3.2 3B Instruct Q8_0 acceptance path visible as a guarded target card until the GGUF exists locally, backend evidence lands, and the load path stays inside Camelid's CPU materialization budget guard
- sends non-streaming chat requests to `POST /v1/chat/completions`
- blocks chat until `/v1/health` reports the selected `active_model_id` with `generation_ready: true` and `/api/capabilities` has an exact supported model/quant compatibility row

Server features Camelid does not expose yet are kept honest: catalog downloads, external-provider setup, planned/future/blocked quantization lanes, and unsupported or partial API parameters show disabled or typed-guardrail copy instead of pretending to work. The analytics view also treats conversation telemetry as usage only, not compatibility evidence. The UI mirrors the compatibility contract documented in `../COMPATIBILITY.md`; filenames, catalog metadata, saved browser paths, and prior usage are not treated as support evidence by themselves.

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

For the exact Llama 3.2 3B Instruct Q8_0 acceptance target, use the exact local path once backend/QA have produced parity evidence, `/api/capabilities` includes an exact supported 3B Q8_0 compatibility row, and the backend no longer fails closed on the CPU materialization budget guard for that model:

```bash
cd frontend
npm run smoke -- --model '$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf' --model-id llama-3.2-3b-instruct-q8 --require-generation
```

Until those conditions are true, this command should either fail at load/generation readiness, return a typed `cpu_weight_materialization_exceeds_budget` guardrail, or skip chat with the explicit support-contract guardrail.
