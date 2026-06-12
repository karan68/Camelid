# Backend asks — endpoint/contract needs discovered during the frontend overhaul

The UI never stubs fake data for these; each ask names the surface that stays
guarded until the contract grows.

## 1. Sampling-parameter capability rows (Phase 2, 2026-06-12)

**Surface waiting on it:** the chat "Generation controls" drawer renders every
sampling parameter (temperature, top_p, top_k, stop, seed) as a guarded
read-only row because `/api/capabilities` `api_features` advertises no sampling
rows at all. Chat keeps sending greedy `temperature: 0` — the lane the parity
evidence covers.

**Ask:** advertise one feature row per parameter the backend actually honors on
`/v1/chat/completions`, with the usual status vocabulary, e.g.:

```json
{
  "id": "sampling_temperature",
  "status": "supported_current_gate",
  "notes": "temperature in [0,2] honored for streaming + non-streaming chat completions; evidence row-scoped to the current gate"
}
```

The frontend already resolves rows by exact id (`sampling_<param>` or
`<param>`, no resemblance matching — `lib/samplingContract.js`) and will unlock
the matching control, persist last-used values per model id, and merge the
override into the request body only while the row stays supported. No frontend
change needed when the rows appear.

**Not asked:** any claim that sampled output has parity evidence. If sampled
lanes need their own evidence categories, that belongs in the compatibility
rows, not the feature row notes.
