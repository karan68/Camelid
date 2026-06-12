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

## 2. Evidence-bundle manifest references on compatibility rows (Phase 4, 2026-06-12)

**Surface waiting on it:** the Compatibility ledger's per-row evidence checklist cites
what the contract exposes today — the `*_pack_id` identifiers (e.g.
`tinyllama-context-512-smoke-v1`). The repo's qa/evidence-bundles manifests
(README/COMPATIBILITY.md reference them by path) are not addressable from
`/api/capabilities`, so chip popovers can name a pack id but cannot cite its manifest.

**Ask:** add an optional manifest reference per evidence lane, e.g.

```json
{
  "bounded_context_512_pack": "validated_bounded_pack",
  "bounded_context_512_pack_id": "tinyllama-context-512-smoke-v1",
  "bounded_context_512_pack_manifest": "qa/evidence-bundles/tinyllama-context-512-.../manifest.json"
}
```

Repo-relative paths only (no absolute filesystem paths — the frontend will render them
as citations, and I7 keeps absolute paths out of shareable surfaces). The ledger picks
up `*_pack_manifest` fields automatically once they appear.
