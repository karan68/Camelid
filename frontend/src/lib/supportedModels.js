/* Curated set of models Camelid supports for local chat (exact rows it knows how
   to download + run). Catalog ids/repos/filenames/sizes match the backend catalog
   the install endpoint expects, so installCatalogModel(item) and the localStorage
   download-progress tracking (keyed by catalog_id) work directly. */
export const SUPPORTED_MODELS = [
  {
    catalog_id: 'llama32_3b_instruct_q8_0',
    name: 'Llama 3.2 3B Instruct',
    repo_id: 'unsloth/Llama-3.2-3B-Instruct-GGUF',
    filename: 'Llama-3.2-3B-Instruct-Q8_0.gguf',
    size_bytes: 3422709216,
    quant: 'Q8_0',
    blurb: 'Reference exact-row model — the best quality/speed balance on Apple Silicon.',
    recommended: true,
  },
  {
    catalog_id: 'llama32_1b_instruct_q8_0',
    name: 'Llama 3.2 1B Instruct',
    repo_id: 'unsloth/Llama-3.2-1B-Instruct-GGUF',
    filename: 'Llama-3.2-1B-Instruct-Q8_0.gguf',
    size_bytes: 1346203104,
    quant: 'Q8_0',
    blurb: 'Smallest supported chat model — fastest to download and load.',
  },
  {
    catalog_id: 'tinyllama_1_1b_chat_q8_0',
    name: 'TinyLlama 1.1B Chat',
    repo_id: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    filename: 'tinyllama-1.1b-chat-v1.0.Q8_0.gguf',
    size_bytes: 1169007424,
    quant: 'Q8_0',
    blurb: 'Tiny SPM-tokenizer chat model — handy for quick smoke checks.',
  },
  {
    catalog_id: 'llama3_8b_instruct_q8_0',
    name: 'Llama 3 8B Instruct',
    repo_id: 'MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF',
    filename: 'Meta-Llama-3-8B-Instruct.Q8_0.gguf',
    size_bytes: 8540846592,
    quant: 'Q8_0',
    blurb: 'Largest supported row — highest quality, needs the most memory.',
  },
  {
    catalog_id: 'gemma4_e4b_it_q8_0',
    name: 'Gemma 4 E4B-It',
    repo_id: 'unsloth/gemma-4-E4B-it-GGUF',
    filename: 'gemma-4-E4B-it-Q8_0.gguf',
    size_bytes: 8192951456,
    quant: 'Q8_0',
    blurb: 'Gemma 4 (E-series matformer) — from-scratch gemma4 engine, greedy-identical to the reference. Serve with CAMELID_GEMMA4_SERVE=1.',
  },
  {
    catalog_id: 'gemma4_e2b_it_q8_0',
    name: 'Gemma 4 E2B-It',
    repo_id: 'unsloth/gemma-4-E2B-it-GGUF',
    filename: 'gemma-4-E2B-it-Q8_0.gguf',
    size_bytes: 5048350848,
    quant: 'Q8_0',
    blurb: 'Gemma 4 E2B (E-series matformer) — greedy parity with the reference on the committed basic_v1 prompt pack. Text-token generation only; serve with CAMELID_GEMMA4_SERVE=1.',
  },
]
