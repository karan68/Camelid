# Ornith-1.0-9B quant quality × residency table (Item 4)

Conductor: ORNITH_9B_CONSTRAINED_VRAM_CONDUCTOR.md Item 4. All quants produced at
REF_QWEN35 (`acd79d6`) from the hash-verified bf16 with
`imatrix_ornith_agentic.gguf` (imatrix over the frozen 20-trace agentic-coding
corpus, 5 chunks @ c=2048, computed on the Q8_0 host — bf16 exceeds host RAM;
deviation documented in RECEIPT_ITEM4).

PPL = llama-perplexity on `heldout_coding.txt` (165 KB repo Rust + llama.cpp C++,
disjoint from calibration), c=2048. Sizes are file bytes. VRAM figures measured on
the RTX 3060 Laptop 6GB (5996 MiB free cold).

## Quality × size

| quant | bytes | bpw | PPL (held-out coding) | vs Q6_K | provenance |
|---|---|---|---|---|---|
| Q6_K | 7,359,259,072 | 6.57 | TBD | — | HF pristine (verifier reference) |
| Q4_K_M (home requant) | 5,629,108,416 | 5.02 | TBD | TBD | Q8_0→Q4 requant, NO imatrix (GPU-lane artifact) |
| IQ4_XS | TBD | TBD | TBD | TBD | bf16 + imatrix (contingency target) |
| Q3_K_M | TBD | TBD | TBD | TBD | bf16 + imatrix |
| IQ3_XXS | TBD | TBD | TBD | TBD | bf16 + imatrix |

## Residency on the 6GB card (16K-context bar: full residency + ≥512 MiB headroom)

Sparse-KV cost at 16,384 positions (8 full-attn layers × 4 kv-heads × 256 dim ×
K+V × f16) ≈ **537 MiB**; DeltaNet state is context-length-independent.

| quant | weights MiB | +KV@16K | headroom @16K (of 5996) | verdict | measured by |
|---|---|---|---|---|---|
| Q4_K_M (home) | 5,368 | ~5,905 | **~90 MiB → FAILS the ≥512 bar** (8K remains the proven config: 5,259 MiB peak, ~700 MiB margin) | fallback row (Item 4 kill-path baseline, already Supported) | TBD empirical |
| IQ4_XS | TBD | TBD | TBD | TBD | llama.cpp `-ngl 99` (Camelid kernels pending) |
| Q3_K_M | TBD | TBD | TBD | TBD | llama.cpp `-ngl 99` (Camelid kernels pending) |
| IQ3_XXS | TBD | TBD | TBD | TBD | llama.cpp `-ngl 99` (Camelid kernels pending) |

## Notes

- Camelid's resident engine implements q8_0/q4_K/q6_K gemv today. The winning
  quant from this table determines which ONE kernel family gets implemented
  (IQ3_XXS/IQ4_XS = new codebook/scale families; Q3_K_M = q3_K + q5_K tensors).
- llama.cpp residency/PPL numbers are guidance for that decision, not Camelid
  promotion evidence; the Camelid residency receipt is minted after the kernel
  lands.
