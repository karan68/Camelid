export const LLAMA32_3B_ACCEPTANCE_TARGET = {
  id: 'llama-3.2-3b-instruct-q8',
  name: 'Llama 3.2 3B Instruct Q8_0',
  model_path: '$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf',
  runtime_model_name: 'llama-3.2-3b-instruct-q8',
  source: 'bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0.gguf',
  provider_kind: 'local',
  status: 'registered',
  engine: 'backendinference',
  quant: 'Q8_0',
  size_gb: '3.19',
  loaded_now: false,
  generation_ready: false,
  backendinference: {
    active: false,
    loaded_now: false,
    generation_ready: false,
    tokenizer_status: null,
    tokenizer_model: null,
    tensor_ready: false,
    config_ready: false,
  },
}

export const LLAMA32_3B_ACCEPTANCE_SUMMARY = 'This exact 3B Q8_0 row is now smoke-supported for short local chat after exact-row /api/models/load, /v1/completions, /v1/chat/completions, frontend smoke, compact parity, and a five-prompt API smoke pack. The claim is intentionally narrow: it does not promote neighboring Llama sizes, other quantizations, the 8B row, longer context, or broad prompt/chat-template behavior.'

export const LLAMA32_3B_ACCEPTANCE_AVAILABILITY = 'This browser/runtime list does not currently show the exact 3B row. That does not erase the existing support evidence for the row, but it also must not be turned into a green frontend state unless the loaded local GGUF exactly matches the supported 3B Q8_0 row.'

export const LLAMA32_3B_ACCEPTANCE_GATING_NOTE = 'Frontend chat unlocks only after Camelid reports loaded_now=true and generation_ready=true for this exact GGUF plus an exact supported Llama 3.2 3B Q8_0 compatibility row; same-family, same-tokenizer, or neighboring-size matches remain blocked.'
