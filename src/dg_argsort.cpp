// DiffusionGemma expert-selection argsort shim (experimental/diffusiongemma).
//
// The reference selects/orders MoE experts with `ggml_argsort_top_k`, whose CPU
// path is libc++ `std::sort` over the expert indices using the STRICT, no-tie-
// break comparator `cmp_argsort<DESC>` = `data[a] > data[b]` (ggml-cpu/ops.cpp).
// `std::sort` is UNSTABLE: for EXACT-equal keys (a true probability tie) the
// relative order is libc++-introsort-internal, NOT lower-index-first. Camelid's
// Rust sort imposed lower-index-first, which swapped two tied experts' slots and
// perturbed the weighted-sum accumulation order (a ~1e-8 seed that the post-norm
// amplified and the stack compounded — block-1 step-3 divergence).
//
// To match BIT-FOR-BIT, we bind the real `std::sort` (same Apple-clang libc++ as
// the pinned reference build) rather than reimplement its introsort — the same
// "bind the real implementation" discipline the lane already uses for
// __sincosf_stret / vDSP. Sorting by the bit-exact router LOGITS with `>` is
// comparison-identical to the reference sorting softmax `selection_probs`
// (softmax is strictly monotonic: logit[a]>logit[b] <=> prob[a]>prob[b], and
// equal logits <=> equal probs), so every comparison std::sort makes is the
// same and the output index order is identical.

#include <algorithm>
#include <cstdint>

extern "C" void dg_argsort_desc_f32(const float * keys, int32_t n, int32_t * out_idx) {
    for (int32_t i = 0; i < n; ++i) {
        out_idx[i] = i;
    }
    // EXACTLY ggml's cmp_argsort<GGML_SORT_ORDER_DESC>: strict `>`, no tie-break.
    std::sort(out_idx, out_idx + n,
              [keys](int32_t a, int32_t b) { return keys[a] > keys[b]; });
}

// ASCENDING variant — matches the reference EB sampler's MI-bound position
// ordering (diffusion.cpp: std::sort(order, entropy[a] < entropy[b]); strict
// `<`, no tie-break, libc++ unstable tie order). Same fix class as the expert
// argsort: Rust's sort_unstable would break entropy ties differently.
extern "C" void dg_argsort_asc_f32(const float * keys, int32_t n, int32_t * out_idx) {
    for (int32_t i = 0; i < n; ++i) {
        out_idx[i] = i;
    }
    std::sort(out_idx, out_idx + n,
              [keys](int32_t a, int32_t b) { return keys[a] < keys[b]; });
}
