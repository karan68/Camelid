# Fixtures

Do not check in large model files. Unit tests generate tiny synthetic GGUF-like files in temp directories. Small licensed fixtures may be added later if needed.

## Tokenizer reference packs

- `tokenizer/tinyllama-chat-template-edge-cases.json` records checked TinyLlama Q8_0 tokenizer/chat-template edge-case reference data for whitespace, multiline turns, control/special characters, and EOS behavior. It is backed by the committed TinyLlama parity bundle and does not promote neighboring rows or broader tokenizer support. Each case keeps llama.cpp's ids (`tokens`) and the model's own SentencePiece ids (`model_tokenizer_tokens`, from `../scripts/gen-tinyllama-sentencepiece-reference.py`). Every token score in this GGUF is 0.0, so llama.cpp's score-ordered merging does not reproduce the model's tokenizer; the tests hold camelid to the SentencePiece ids.
- `tokenizer/llama3-reference-tokenizer.json` records checked Llama 3 tokenizer reference data used by tests.
- `tokenizer/mistral-7b-instruct-v0.3-reference-pack.template.json` is the planning template for the current exact Mistral bring-up row. It is intentionally not evidence and must be filled only with row-specific checked reference data from the exact chosen GGUF.
- `../scripts/mistral_smoke_reference.sh` captures the exact-row Mistral tokenizer/chat-template reference pack with `llama.cpp llama-tokenize` once the chosen GGUF is present.
