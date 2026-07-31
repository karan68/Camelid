//! Sliding-window attention semantics — the single pinned reference.
//!
//! Gemma 3 (and Gemma 4) run *local* decoder layers whose attention is masked to a
//! trailing window of positions. Every implementation in this tree encodes the same
//! convention, and every one of them spells it out separately:
//!
//! | site | expression |
//! |---|---|
//! | Metal resident decode encode (`src/metal.rs`) | `window_start = filled.saturating_sub(window)`, `filled = position + 1` |
//! | Gemma 4 CPU runtime (`src/gemma4_runtime.rs`) | `lo = (pos + 1).saturating_sub(win)` |
//! | `Gemma3Metadata::layer_window` / `Gemma4LayerPlan::window` (`src/model.rs`) | doc: attend `[pos + 1 - window ..= pos]` |
//!
//! **The window INCLUDES the current position.** A layer with `window = 512` at
//! absolute position `p` attends `[max(0, p + 1 - 512) ..= p]` — the current token
//! plus the 511 before it, i.e. exactly `min(p + 1, 512)` positions.
//!
//! This module is that convention written once, so a batched prefill kernel can be
//! checked against a reference instead of against a re-derivation of the same
//! arithmetic. It is deliberately dependency-free and arch-independent: it knows
//! nothing about Metal, gemma3 or gemma4, only about the mask.
//!
//! ## Why the predicate is written as an addition
//!
//! [`is_visible`] evaluates `key + window > query`, never `key >= query - window`.
//! The subtraction underflows for `query < window` on unsigned types (the exact bug
//! class a batched kernel introduces when the mask moves from the host into MSL,
//! where there is no `saturating_sub`), and the `>` rather than `>=` is where the
//! INCLUDES-current-position convention lives:
//!
//! - at `key = query + 1 - window` the predicate is `query + 1 > query` → visible
//!   (the oldest in-window position);
//! - at `key = query - window` it is `query > query` → hidden (the first position
//!   outside).
//!
//! [`window_bounds`] and [`is_visible`] are cross-checked against each other, and
//! against the two production expressions above, by this module's unit tests.

use std::ops::RangeInclusive;

/// Resolve a sliding window to the `[lo, lo + count)` slice of key positions that
/// query position `position` attends.
///
/// `window == None` (a global / full-causal layer) yields `lo = 0`, i.e. the whole
/// causal prefix. `window == Some(0)` is degenerate and is treated as full causal,
/// matching the "0 = disabled" uniform convention the batched kernels will use.
///
/// Returns `(window_start, position_count)` — deliberately the same names and the
/// same two values the Metal resident decode encode computes and hands to
/// `encode_attention`.
#[must_use]
pub fn window_bounds(position: usize, window: Option<usize>) -> (usize, usize) {
    let filled = position + 1;
    let start = match window {
        Some(w) if w > 0 => filled.saturating_sub(w),
        _ => 0,
    };
    (start, filled - start)
}

/// The full per-element attention mask predicate: is key position `key` visible to
/// query position `query` under `window`?
///
/// Causal upper bound and windowed lower bound in one expression, in the
/// unsigned-safe form a batched kernel must use.
#[must_use]
pub fn is_visible(query: usize, key: usize, window: Option<usize>) -> bool {
    if key > query {
        return false;
    }
    match window {
        Some(w) if w > 0 => key + w > query,
        _ => true,
    }
}

/// The inclusive range of key positions visible to `query`.
#[must_use]
pub fn visible_range(query: usize, window: Option<usize>) -> RangeInclusive<usize> {
    let (lo, _) = window_bounds(query, window);
    lo..=query
}

/// Number of key positions visible to `query` — `min(query + 1, window)`.
#[must_use]
pub fn visible_count(query: usize, window: Option<usize>) -> usize {
    window_bounds(query, window).1
}

/// Reference windowed single-query attention, f32, no fast-math re-association.
///
/// `q` is one query head (`head_dim` values). `k` and `v` are that head's cache in
/// `[position][head_dim]` order, covering positions `0..=query`. The softmax is the
/// standard max-subtracted form; the reduction order is source order, so this is a
/// *semantic* reference (it pins which positions are summed), not a bit-exact model
/// of any kernel's reduction tree.
///
/// Panics if the caches are shorter than `query + 1` positions.
#[must_use]
pub fn windowed_attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    query: usize,
    window: Option<usize>,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), head_dim, "query row must be head_dim wide");
    assert!(
        k.len() >= (query + 1) * head_dim && v.len() >= (query + 1) * head_dim,
        "K/V cache must cover positions 0..={query}"
    );
    let range = visible_range(query, window);
    let mut scores: Vec<f32> = range
        .clone()
        .map(|p| {
            let row = &k[p * head_dim..(p + 1) * head_dim];
            q.iter().zip(row).map(|(a, b)| a * b).sum::<f32>() * scale
        })
        .collect();
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut den = 0.0f32;
    for s in &mut scores {
        *s = (*s - m).exp();
        den += *s;
    }
    let inv = if den == 0.0 { 0.0 } else { 1.0 / den };
    let mut out = vec![0.0f32; head_dim];
    for (i, p) in range.enumerate() {
        let w = scores[i] * inv;
        let row = &v[p * head_dim..(p + 1) * head_dim];
        for (o, vv) in out.iter_mut().zip(row) {
            *o += w * vv;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim re-statement of the Metal resident decode encode
    /// (`src/metal.rs`: `let filled = position + 1;` …
    /// `window_start = filled.saturating_sub(w)`, `position_count = filled - window_start`).
    fn metal_encode_expression(position: usize, window: Option<usize>) -> (usize, usize) {
        let filled = position + 1;
        let window_start = window.map_or(0, |w| filled.saturating_sub(w));
        (window_start, filled - window_start)
    }

    /// Verbatim re-statement of the Gemma 4 CPU runtime
    /// (`src/gemma4_runtime.rs`: `let lo = if sliding { (pos + 1).saturating_sub(win) } else { 0 };`
    /// then `scores: (lo..=pos)`).
    fn gemma4_cpu_expression(pos: usize, window: Option<usize>) -> RangeInclusive<usize> {
        let lo = match window {
            Some(win) => (pos + 1).saturating_sub(win),
            None => 0,
        };
        lo..=pos
    }

    #[test]
    fn window_bounds_matches_the_metal_encode_and_the_gemma4_cpu_path() {
        for window in [None, Some(1), Some(2), Some(4), Some(512), Some(1024)] {
            for position in 0..2600usize {
                assert_eq!(
                    window_bounds(position, window),
                    metal_encode_expression(position, window),
                    "position {position} window {window:?}: diverged from the Metal encode"
                );
                let r = visible_range(position, window);
                let g4 = gemma4_cpu_expression(position, window);
                assert_eq!(
                    (*r.start(), *r.end()),
                    (*g4.start(), *g4.end()),
                    "position {position} window {window:?}: diverged from the gemma4 CPU path"
                );
            }
        }
    }

    #[test]
    fn is_visible_agrees_with_window_bounds_elementwise() {
        for window in [None, Some(0), Some(1), Some(3), Some(512)] {
            for query in 0..1100usize {
                let (lo, count) = window_bounds(query, window);
                for key in 0..1200usize {
                    let expected = key >= lo && key <= query;
                    assert_eq!(
                        is_visible(query, key, window),
                        expected,
                        "query {query} key {key} window {window:?}"
                    );
                }
                assert_eq!(count, query - lo + 1, "count must match the resolved range");
            }
        }
    }

    /// The three positions the window-edge prompt pack is built around, spelled out
    /// so an off-by-one anywhere in the campaign fails HERE first.
    #[test]
    fn window_512_edge_is_q_minus_511_inside_and_q_minus_512_outside() {
        let w = Some(512);
        for q in [512usize, 1023, 1024, 2047, 2399] {
            assert!(
                is_visible(q, q - 511, w),
                "q-511 must be the OLDEST visible position (q = {q})"
            );
            assert!(
                !is_visible(q, q - 512, w),
                "q-512 must be the FIRST position outside the window (q = {q})"
            );
            assert!(
                is_visible(q, q, w),
                "the current position is always visible"
            );
            assert_eq!(visible_count(q, w), 512);
            assert_eq!(*visible_range(q, w).start(), q - 511);
        }
        // Below the window nothing is clipped, so an off-by-one on the window value
        // itself is INVISIBLE here — the pack must exceed 512 to have power.
        for q in 0..511usize {
            assert_eq!(window_bounds(q, Some(512)), window_bounds(q, Some(513)));
            assert_eq!(visible_count(q, w), q + 1);
        }
        assert_eq!(window_bounds(511, Some(512)), (0, 512));
        assert_eq!(window_bounds(511, Some(511)), (1, 511));
    }

    #[test]
    fn window_zero_and_none_are_the_same_full_causal_mask() {
        for q in [0usize, 1, 63, 64, 511, 512, 2400] {
            assert_eq!(window_bounds(q, None), window_bounds(q, Some(0)));
            assert_eq!(window_bounds(q, None), (0, q + 1));
            for key in 0..=q {
                assert!(is_visible(q, key, None));
                assert!(is_visible(q, key, Some(0)));
            }
        }
    }

    #[test]
    fn reference_attention_ignores_positions_outside_the_window() {
        let head_dim = 4usize;
        let query = 5usize;
        let window = Some(3);
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut k = vec![0.0f32; (query + 1) * head_dim];
        let mut v = vec![0.0f32; (query + 1) * head_dim];
        for p in 0..=query {
            k[p * head_dim] = 1.0;
            v[p * head_dim] = p as f32;
        }
        // Positions 0..=2 are outside the window and must not contribute; the three
        // in-window V rows are 3, 4, 5 with equal scores -> mean 4.0.
        let out = windowed_attention_reference(&q, &k, &v, head_dim, query, window, 1.0);
        assert!((out[0] - 4.0).abs() < 1e-6, "got {out:?}");
        // Poisoning an out-of-window V row must not move the answer.
        v[0] = 1.0e6;
        let out2 = windowed_attention_reference(&q, &k, &v, head_dim, query, window, 1.0);
        assert_eq!(out[0].to_bits(), out2[0].to_bits());
        // Full-causal over the same data DOES move: 0..=5 mean = 2.5.
        let full = windowed_attention_reference(&q, &k, &v, head_dim, query, None, 1.0);
        assert!(
            full[0] > 1.0e4,
            "full-causal must see the poisoned row: {full:?}"
        );
    }
}
