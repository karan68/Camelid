# Fixtures

Do not check in large model files. Unit tests generate tiny synthetic GGUF-like files in temp directories. Small licensed fixtures may be added later if needed.

## Tokenizer reference packs

- `tokenizer/llama3-reference-tokenizer.json` records checked Llama 3 tokenizer reference data used by tests.
- `tokenizer/mistral-7b-instruct-v0.2-reference-pack.template.json` is a planning template for the first exact Mistral bring-up row. It is intentionally not evidence and must be filled only with row-specific checked reference data from the exact chosen GGUF.
