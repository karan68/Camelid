//! KV-cache equivalence — the direct invariant for batched prefill.
//!
//! End-to-end token identity is the *outer* gate on a batched prefill and a poor
//! *inner* one: it observes one argmax per step, after 26 layers of mixing, and the
//! campaign's own measurements bound the dead zone it cannot see. Reduction-order
//! noise on this row is 2.122e-4 max |logit diff|; the smallest observed argmax flip
//! sits at a 0.0032-nat top-2 gap. A perturbation between those bounds — precisely
//! what an off-by-one window bound or a wrong row stride produces at a handful of
//! positions — is invisible to any argmax-only check.
//!
//! This module is the cheap direct alternative: a batched prefill of `n` tokens must
//! leave the KV cache in **the same state** as `n` token-by-token forwards. That is
//! a per-(layer, position, head, dim) claim over ~10^8 values with no softmax between
//! the defect and the observable, so an error that moves one K row by 1e-6 is caught
//! at that row instead of being averaged away.
//!
//! ## The two tiers, and why they are different assertions
//!
//! - **Tier A (batched weight streaming)** keeps every reduction serial in the same
//!   order as the single-token path — batching over the token dimension re-associates
//!   nothing — so its KV must be **bit-identical**. Use [`KvEquivalence::assert_bit_identical`].
//!   There is no tolerance to negotiate here; a differing bit is a bug.
//! - **Tier B (tiled/batched attention)** changes the reduction order inside
//!   attention, so its KV is *not* bit-identical by construction. Use
//!   [`KvEquivalence::meets_bound`] with a bound **published before the run**, plus the
//!   outlier check: a uniform small delta across every position is reduction noise, a
//!   single position 10x above the median is a mask or stride defect wearing noise as
//!   a disguise. The outlier check is the part that actually has power; the scalar
//!   bound alone can be met by a wrong kernel.
//!
//! Neither tier's snapshot is expensive: the caches are already in shared-storage
//! Metal buffers, so a snapshot is a memcpy plus a format widen.
//!
//! ## Where a defect becomes visible — and the one place the caches cannot see
//!
//! Layer 0's K/V comes from the token embedding, never from attention, so a mask
//! defect is invisible in layer 0's cache and first shows in layer 1's. By the same
//! argument the **last** layer's attention output projects no K/V at all: a defect
//! confined to it moves zero cache elements. That is why a snapshot carries
//! `final_hidden` and why [`KvEquivalence::meets_bound`] bounds it separately —
//! measured, not hypothesized: the `window_on_all_layers` mutant in
//! `metal_kv_snapshot_equivalence_catches_window_and_rope_mutations` leaves every
//! cache bit intact and moves 1 152 hidden elements, and it survived an earlier
//! draft of `meets_bound` that checked only the caches.
//!
//! ## Canonical layout
//!
//! A snapshot stores, per layer, `[kv_head][position][head_dim]` contiguous f32 — the
//! layout `ResidentDecodeState::read_from` already produces, with the per-head
//! `max_positions` stride removed so two sessions with different capacities compare
//! cleanly. [`KvSnapshot::digest`] hashes that canonical form, so a digest is
//! comparable across processes and can be committed in a receipt.

use sha2::{Digest, Sha256};

/// Which tensor a difference was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTensor {
    /// The rotated, per-head-normed K cache.
    Key,
    /// The V cache.
    Value,
    /// The final hidden state after the last layer (not a cache, but the same claim).
    FinalHidden,
}

impl KvTensor {
    /// Short label for assertion messages and receipts.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            KvTensor::Key => "K",
            KvTensor::Value => "V",
            KvTensor::FinalHidden => "final_hidden",
        }
    }
}

/// Cache geometry a snapshot was taken at. Two snapshots only compare when these
/// match exactly — a geometry mismatch is a harness bug, never a parity result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvGeometry {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Number of cached positions captured, counting from 0.
    pub positions: usize,
}

impl KvGeometry {
    /// Elements per layer in one cache tensor.
    #[must_use]
    pub fn per_layer_elements(&self) -> usize {
        self.n_kv_heads * self.positions * self.head_dim
    }

    /// Index of `(kv_head, position, dim)` inside one layer's contiguous cache.
    #[must_use]
    pub fn index(&self, kv_head: usize, position: usize, dim: usize) -> usize {
        (kv_head * self.positions + position) * self.head_dim + dim
    }
}

/// A point-in-time copy of a resident session's KV cache (and optionally the final
/// hidden state), in the canonical `[kv_head][position][head_dim]` layout.
#[derive(Debug, Clone, PartialEq)]
pub struct KvSnapshot {
    pub geometry: KvGeometry,
    /// `k[layer]` is `per_layer_elements()` f32 values.
    pub k: Vec<Vec<f32>>,
    /// `v[layer]` is `per_layer_elements()` f32 values.
    pub v: Vec<Vec<f32>>,
    /// The hidden state emitted by the last forward, when the harness captured one.
    pub final_hidden: Option<Vec<f32>>,
}

/// Why two snapshots could not be compared at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCompareError {
    /// Geometries differ; the comparison is meaningless.
    GeometryMismatch { left: KvGeometry, right: KvGeometry },
    /// A snapshot's buffers do not match its own declared geometry.
    Malformed(&'static str),
    /// One side captured a final hidden state and the other did not.
    HiddenPresenceMismatch,
    /// Both captured a final hidden state, of different lengths.
    HiddenLengthMismatch { left: usize, right: usize },
}

impl std::fmt::Display for KvCompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCompareError::GeometryMismatch { left, right } => {
                write!(f, "KV geometry mismatch: {left:?} vs {right:?}")
            }
            KvCompareError::Malformed(what) => write!(f, "malformed KV snapshot: {what}"),
            KvCompareError::HiddenPresenceMismatch => {
                write!(
                    f,
                    "one snapshot carries a final hidden state and the other does not"
                )
            }
            KvCompareError::HiddenLengthMismatch { left, right } => {
                write!(f, "final hidden length mismatch: {left} vs {right}")
            }
        }
    }
}

impl std::error::Error for KvCompareError {}

impl KvSnapshot {
    /// An all-zero snapshot at `geometry`, to be filled by a capture hook.
    #[must_use]
    pub fn zeroed(geometry: KvGeometry) -> Self {
        let per = geometry.per_layer_elements();
        Self {
            geometry,
            k: vec![vec![0.0; per]; geometry.n_layers],
            v: vec![vec![0.0; per]; geometry.n_layers],
            final_hidden: None,
        }
    }

    /// Build from per-layer `(k, v)` buffers already in canonical layout.
    ///
    /// Returns `Err` if any buffer's length disagrees with `geometry`.
    pub fn from_layers(
        geometry: KvGeometry,
        layers: Vec<(Vec<f32>, Vec<f32>)>,
    ) -> Result<Self, KvCompareError> {
        if layers.len() != geometry.n_layers {
            return Err(KvCompareError::Malformed("layer count"));
        }
        let per = geometry.per_layer_elements();
        let mut k = Vec::with_capacity(layers.len());
        let mut v = Vec::with_capacity(layers.len());
        for (lk, lv) in layers {
            if lk.len() != per || lv.len() != per {
                return Err(KvCompareError::Malformed("per-layer element count"));
            }
            k.push(lk);
            v.push(lv);
        }
        Ok(Self {
            geometry,
            k,
            v,
            final_hidden: None,
        })
    }

    /// Attach the final hidden state (builder style).
    #[must_use]
    pub fn with_final_hidden(mut self, hidden: Vec<f32>) -> Self {
        self.final_hidden = Some(hidden);
        self
    }

    /// One `(layer, position)` K row across all KV heads, `n_kv_heads * head_dim` values.
    #[must_use]
    pub fn key_row(&self, layer: usize, position: usize) -> Vec<f32> {
        self.row(&self.k[layer], position)
    }

    /// One `(layer, position)` V row across all KV heads.
    #[must_use]
    pub fn value_row(&self, layer: usize, position: usize) -> Vec<f32> {
        self.row(&self.v[layer], position)
    }

    fn row(&self, buf: &[f32], position: usize) -> Vec<f32> {
        let g = &self.geometry;
        let mut out = Vec::with_capacity(g.n_kv_heads * g.head_dim);
        for h in 0..g.n_kv_heads {
            let base = g.index(h, position, 0);
            out.extend_from_slice(&buf[base..base + g.head_dim]);
        }
        out
    }

    /// SHA-256 over the canonical serialization: geometry (four u64 LE), then for each
    /// layer K then V as raw f32 bits LE, then a presence byte and the final hidden.
    ///
    /// Raw *bits*, not values: `-0.0` and `+0.0` must not hash alike, and a NaN payload
    /// change is a real change.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        let g = &self.geometry;
        for field in [g.n_layers, g.n_kv_heads, g.head_dim, g.positions] {
            h.update((field as u64).to_le_bytes());
        }
        for layer in 0..g.n_layers {
            for buf in [&self.k[layer], &self.v[layer]] {
                for value in buf {
                    h.update(value.to_bits().to_le_bytes());
                }
            }
        }
        match &self.final_hidden {
            Some(hidden) => {
                h.update([1u8]);
                h.update((hidden.len() as u64).to_le_bytes());
                for value in hidden {
                    h.update(value.to_bits().to_le_bytes());
                }
            }
            None => h.update([0u8]),
        }
        format!("{:x}", h.finalize())
    }
}

/// Where a difference sits, and what the two sides held there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvDiffSite {
    pub tensor: KvTensor,
    pub layer: usize,
    pub kv_head: usize,
    pub position: usize,
    pub dim: usize,
    pub left: f32,
    pub right: f32,
}

impl std::fmt::Display for KvDiffSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.tensor == KvTensor::FinalHidden {
            return write!(
                f,
                "final_hidden[{}]: {} (0x{:08x}) vs {} (0x{:08x})",
                self.dim,
                self.left,
                self.left.to_bits(),
                self.right,
                self.right.to_bits()
            );
        }
        write!(
            f,
            "{} layer {} kv_head {} position {} dim {}: {} (0x{:08x}) vs {} (0x{:08x})",
            self.tensor.label(),
            self.layer,
            self.kv_head,
            self.position,
            self.dim,
            self.left,
            self.left.to_bits(),
            self.right,
            self.right.to_bits()
        )
    }
}

/// The verdict of comparing two KV snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct KvEquivalence {
    pub geometry: KvGeometry,
    /// True when EVERY compared f32 has identical bits, including the final hidden.
    pub bit_identical: bool,
    /// Count of elements whose bits differ.
    pub differing_elements: usize,
    /// The first differing element in canonical scan order (layer, K then V, head,
    /// position, dim) — the one to print when a bit gate fails.
    pub first_difference: Option<KvDiffSite>,
    /// Largest |a - b| over the caches. `f32::INFINITY` if either side held a NaN
    /// where the other did not (a NaN never "matches within a bound").
    pub max_abs_diff: f32,
    /// Where `max_abs_diff` was attained.
    pub max_abs_diff_site: Option<KvDiffSite>,
    /// Per-position max |a - b| over every layer, head, dim and both caches. Length
    /// `geometry.positions`. This is the vector the outlier test runs on.
    pub per_position_max_abs: Vec<f32>,
    /// Per-layer max |a - b|. Length `geometry.n_layers`.
    pub per_layer_max_abs: Vec<f32>,
    /// Final-hidden bit equality (true when neither side captured one).
    pub hidden_bit_identical: bool,
    /// Final-hidden max |a - b| (0.0 when neither side captured one).
    pub hidden_max_abs_diff: f32,
}

fn diff(a: f32, b: f32) -> f32 {
    if a.to_bits() == b.to_bits() {
        return 0.0;
    }
    if a.is_nan() || b.is_nan() {
        return f32::INFINITY;
    }
    (a - b).abs()
}

/// Compare two KV snapshots taken at the same geometry.
///
/// Scans in canonical order so `first_difference` is deterministic and reproducible
/// across runs and machines.
pub fn compare(left: &KvSnapshot, right: &KvSnapshot) -> Result<KvEquivalence, KvCompareError> {
    let g = left.geometry;
    if g != right.geometry {
        return Err(KvCompareError::GeometryMismatch {
            left: g,
            right: right.geometry,
        });
    }
    let per = g.per_layer_elements();
    for snap in [left, right] {
        if snap.k.len() != g.n_layers || snap.v.len() != g.n_layers {
            return Err(KvCompareError::Malformed("layer count"));
        }
        if snap.k.iter().chain(&snap.v).any(|b| b.len() != per) {
            return Err(KvCompareError::Malformed("per-layer element count"));
        }
    }

    let mut out = KvEquivalence {
        geometry: g,
        bit_identical: true,
        differing_elements: 0,
        first_difference: None,
        max_abs_diff: 0.0,
        max_abs_diff_site: None,
        per_position_max_abs: vec![0.0; g.positions],
        per_layer_max_abs: vec![0.0; g.n_layers],
        hidden_bit_identical: true,
        hidden_max_abs_diff: 0.0,
    };

    for layer in 0..g.n_layers {
        for (tensor, lb, rb) in [
            (KvTensor::Key, &left.k[layer], &right.k[layer]),
            (KvTensor::Value, &left.v[layer], &right.v[layer]),
        ] {
            for kv_head in 0..g.n_kv_heads {
                for position in 0..g.positions {
                    let base = g.index(kv_head, position, 0);
                    for dim in 0..g.head_dim {
                        let a = lb[base + dim];
                        let b = rb[base + dim];
                        if a.to_bits() == b.to_bits() {
                            continue;
                        }
                        let site = KvDiffSite {
                            tensor,
                            layer,
                            kv_head,
                            position,
                            dim,
                            left: a,
                            right: b,
                        };
                        out.bit_identical = false;
                        out.differing_elements += 1;
                        if out.first_difference.is_none() {
                            out.first_difference = Some(site);
                        }
                        let d = diff(a, b);
                        if d > out.max_abs_diff {
                            out.max_abs_diff = d;
                            out.max_abs_diff_site = Some(site);
                        }
                        if d > out.per_position_max_abs[position] {
                            out.per_position_max_abs[position] = d;
                        }
                        if d > out.per_layer_max_abs[layer] {
                            out.per_layer_max_abs[layer] = d;
                        }
                    }
                }
            }
        }
    }

    match (&left.final_hidden, &right.final_hidden) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            if a.len() != b.len() {
                return Err(KvCompareError::HiddenLengthMismatch {
                    left: a.len(),
                    right: b.len(),
                });
            }
            for (dim, (x, y)) in a.iter().zip(b).enumerate() {
                if x.to_bits() == y.to_bits() {
                    continue;
                }
                out.hidden_bit_identical = false;
                out.bit_identical = false;
                out.differing_elements += 1;
                let site = KvDiffSite {
                    tensor: KvTensor::FinalHidden,
                    layer: g.n_layers,
                    kv_head: 0,
                    position: g.positions.saturating_sub(1),
                    dim,
                    left: *x,
                    right: *y,
                };
                if out.first_difference.is_none() {
                    out.first_difference = Some(site);
                }
                let d = diff(*x, *y);
                if d > out.hidden_max_abs_diff {
                    out.hidden_max_abs_diff = d;
                }
            }
        }
        _ => return Err(KvCompareError::HiddenPresenceMismatch),
    }

    Ok(out)
}

impl KvEquivalence {
    /// **G1 — Tier A.** Panic unless every compared bit is identical.
    ///
    /// `context` is prefixed to the message so a sweep over prompt lengths says which
    /// length failed.
    pub fn assert_bit_identical(&self, context: &str) {
        if self.bit_identical {
            return;
        }
        let site = self
            .first_difference
            .expect("a non-bit-identical verdict always carries a first difference");
        panic!(
            "{context}: batched KV is NOT bit-identical to token-by-token — \
             {} differing element(s) of {}, first at {site}, max |diff| {:.6e}. \
             Tier A re-associates no reduction; a differing bit is a bug, not a tolerance.",
            self.differing_elements,
            self.compared_elements(),
            self.max_abs_diff
        );
    }

    /// Total f32 elements the comparison covered (caches only).
    #[must_use]
    pub fn compared_elements(&self) -> usize {
        2 * self.geometry.n_layers * self.geometry.per_layer_elements()
    }

    /// Median of `per_position_max_abs`. Positions with no difference count as 0, so a
    /// defect at a handful of positions leaves the median at (or near) zero and the
    /// outlier test fires immediately.
    #[must_use]
    pub fn median_position_max_abs(&self) -> f32 {
        if self.per_position_max_abs.is_empty() {
            return 0.0;
        }
        let mut v = self.per_position_max_abs.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    }

    /// Positions whose max |diff| exceeds `factor x` the median.
    ///
    /// When the median is exactly 0 (the common case for a localized defect) every
    /// differing position is an outlier — which is the intent.
    #[must_use]
    pub fn outlier_positions(&self, factor: f32) -> Vec<usize> {
        let median = self.median_position_max_abs();
        let threshold = if median > 0.0 { median * factor } else { 0.0 };
        self.per_position_max_abs
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// **G6 — Tier B.** `Ok(())` when the scalar bound holds AND no position is an
    /// outlier by more than `outlier_factor` times the median.
    ///
    /// Both halves are load-bearing. The scalar bound alone is satisfiable by a kernel
    /// that is uniformly slightly wrong; the outlier half is what catches a mask or
    /// stride defect that touches few positions hard. The final hidden is bounded by
    /// the same value, separately — see the comment inside.
    pub fn meets_bound(&self, bound: f32, outlier_factor: f32) -> Result<(), String> {
        // `>` is safe rather than `!(<=)`: `diff()` maps any NaN-vs-non-NaN pair to
        // `f32::INFINITY`, so these two fields are never NaN and the comparison is
        // total. An infinite delta therefore fails every finite bound, which is the
        // intent.
        if self.max_abs_diff > bound {
            let site = self
                .max_abs_diff_site
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<none>".to_string());
            return Err(format!(
                "max |KV diff| {:.6e} exceeds the published bound {bound:.6e} at {site}",
                self.max_abs_diff
            ));
        }
        // The final hidden is checked SEPARATELY and is not optional. The LAST
        // layer's attention output reaches no KV cache — there is no layer after it
        // to project K/V from — so a mask defect confined to the last layer moves
        // zero cache elements and is visible ONLY here. (Measured: the
        // `window_on_all_layers` mutant in
        // `metal_kv_snapshot_equivalence_catches_window_and_rope_mutations` leaves
        // every cache bit intact and moves 1 152 hidden elements.) A bound check
        // that skipped the hidden would pass that mutant.
        if self.hidden_max_abs_diff > bound {
            return Err(format!(
                "max |final_hidden diff| {:.6e} exceeds the published bound {bound:.6e} \
                 (caches agree to {:.6e}) — a defect confined to the LAST layer's attention \
                 reaches no KV cache and is visible only here",
                self.hidden_max_abs_diff, self.max_abs_diff
            ));
        }
        if !self.hidden_bit_identical && self.hidden_max_abs_diff == 0.0 {
            return Err(
                "final_hidden differs in BITS at zero absolute distance (signed zero or a NaN \
                 payload change) — not reduction noise"
                    .to_string(),
            );
        }
        let outliers = self.outlier_positions(outlier_factor);
        if !outliers.is_empty() {
            let median = self.median_position_max_abs();
            let head: Vec<String> = outliers
                .iter()
                .take(8)
                .map(|&p| format!("{p} ({:.3e})", self.per_position_max_abs[p]))
                .collect();
            return Err(format!(
                "{} position(s) exceed {outlier_factor}x the per-position median {median:.6e} \
                 — a localized defect, not reduction noise. First: {}",
                outliers.len(),
                head.join(", ")
            ));
        }
        Ok(())
    }

    /// One-line summary for a receipt or a `--nocapture` log.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "bit_identical={} differing={}/{} kv_max_abs={:.6e} median_pos_max={:.6e} \
             hidden_bit_identical={} hidden_max_abs={:.6e}",
            self.bit_identical,
            self.differing_elements,
            self.compared_elements(),
            self.max_abs_diff,
            self.median_position_max_abs(),
            self.hidden_bit_identical,
            self.hidden_max_abs_diff
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> KvGeometry {
        KvGeometry {
            n_layers: 3,
            n_kv_heads: 2,
            head_dim: 4,
            positions: 6,
        }
    }

    fn filled(g: KvGeometry, seed: f32) -> KvSnapshot {
        let per = g.per_layer_elements();
        let mut s = KvSnapshot::zeroed(g);
        for layer in 0..g.n_layers {
            for i in 0..per {
                s.k[layer][i] = seed + (layer * per + i) as f32 * 0.25;
                s.v[layer][i] = seed - (layer * per + i) as f32 * 0.125;
            }
        }
        s
    }

    #[test]
    fn identical_snapshots_are_bit_identical_and_share_a_digest() {
        let g = geom();
        let a = filled(g, 1.0);
        let b = filled(g, 1.0);
        let v = compare(&a, &b).expect("compare");
        assert!(v.bit_identical);
        assert_eq!(v.differing_elements, 0);
        assert_eq!(v.max_abs_diff, 0.0);
        assert_eq!(a.digest(), b.digest());
        v.assert_bit_identical("identity");
        assert!(v.meets_bound(0.0, 10.0).is_ok());
    }

    #[test]
    fn one_flipped_bit_is_located_exactly_and_fails_the_tier_a_gate() {
        let g = geom();
        let a = filled(g, 1.0);
        let mut b = a.clone();
        // layer 1, kv_head 1, position 3, dim 2
        let idx = g.index(1, 3, 2);
        b.k[1][idx] = f32::from_bits(b.k[1][idx].to_bits() ^ 1);
        let v = compare(&a, &b).expect("compare");
        assert!(!v.bit_identical);
        assert_eq!(v.differing_elements, 1);
        let site = v.first_difference.expect("site");
        assert_eq!(site.tensor, KvTensor::Key);
        assert_eq!(
            (site.layer, site.kv_head, site.position, site.dim),
            (1, 1, 3, 2)
        );
        assert_ne!(a.digest(), b.digest());
        // The outlier test fires on a single-position defect even though the absolute
        // magnitude is a single ULP.
        assert_eq!(v.outlier_positions(10.0), vec![3]);
        assert!(v.meets_bound(1.0, 10.0).is_err());
    }

    #[test]
    fn assert_bit_identical_panics_with_the_site() {
        let g = geom();
        let a = filled(g, 1.0);
        let mut b = a.clone();
        b.v[2][g.index(0, 5, 1)] += 1.0;
        let v = compare(&a, &b).expect("compare");
        let err = std::panic::catch_unwind(|| v.assert_bit_identical("N=513")).unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| "<non-string panic>".to_string());
        assert!(msg.contains("N=513"), "{msg}");
        assert!(msg.contains("position 5"), "{msg}");
        assert!(msg.contains("V layer 2"), "{msg}");
    }

    #[test]
    fn uniform_noise_passes_the_bound_but_a_localized_spike_does_not() {
        let g = geom();
        let a = filled(g, 1.0);
        // Uniform 1e-6-scale perturbation everywhere: passes bound AND outlier test,
        // because every position moves by the same amount.
        let mut uniform = a.clone();
        for layer in 0..g.n_layers {
            for i in 0..g.per_layer_elements() {
                uniform.k[layer][i] += 1.0e-6;
                uniform.v[layer][i] += 1.0e-6;
            }
        }
        let v = compare(&a, &uniform).expect("compare");
        assert!(
            !v.bit_identical,
            "a tolerance-scale change is still not bit-identical"
        );
        assert!(
            v.meets_bound(1.0e-5, 10.0).is_ok(),
            "uniform reduction-order noise must pass: {:?}",
            v.meets_bound(1.0e-5, 10.0)
        );
        // Same scalar bound, but concentrated at one position: must FAIL.
        let mut spike = a.clone();
        for layer in 0..g.n_layers {
            for h in 0..g.n_kv_heads {
                for d in 0..g.head_dim {
                    spike.k[layer][g.index(h, 4, d)] += 1.0e-6;
                }
            }
        }
        let v2 = compare(&a, &spike).expect("compare");
        assert!(v2.max_abs_diff <= 1.0e-5);
        let err = v2
            .meets_bound(1.0e-5, 10.0)
            .expect_err("localized spike must fail");
        assert!(err.contains("localized defect"), "{err}");
        assert_eq!(v2.outlier_positions(10.0), vec![4]);
    }

    #[test]
    fn nan_never_matches_within_a_bound() {
        let g = geom();
        let a = filled(g, 1.0);
        let mut b = a.clone();
        b.k[0][0] = f32::NAN;
        let v = compare(&a, &b).expect("compare");
        assert!(!v.bit_identical);
        assert!(v.max_abs_diff.is_infinite());
        assert!(v.meets_bound(f32::MAX, 1.0e9).is_err());
    }

    #[test]
    fn final_hidden_participates_in_both_the_digest_and_the_verdict() {
        let g = geom();
        let a = filled(g, 1.0).with_final_hidden(vec![1.0, 2.0, 3.0]);
        let b = filled(g, 1.0).with_final_hidden(vec![1.0, 2.0, 3.0]);
        let v = compare(&a, &b).expect("compare");
        assert!(v.bit_identical && v.hidden_bit_identical);
        assert_eq!(a.digest(), b.digest());

        let c = filled(g, 1.0).with_final_hidden(vec![1.0, 2.0, 3.5]);
        let v2 = compare(&a, &c).expect("compare");
        assert!(!v2.bit_identical);
        assert!(!v2.hidden_bit_identical);
        assert!((v2.hidden_max_abs_diff - 0.5).abs() < 1e-6);
        assert_ne!(a.digest(), c.digest());

        // Presence mismatch is an error, not a "difference".
        let d = filled(g, 1.0);
        assert_eq!(
            compare(&a, &d).unwrap_err(),
            KvCompareError::HiddenPresenceMismatch
        );
    }

    /// The last layer's attention output projects no K/V, so a defect confined to it
    /// moves zero cache elements. `meets_bound` must still fail. This is not a
    /// hypothetical: `metal_kv_snapshot_equivalence_catches_window_and_rope_mutations`
    /// produced exactly this shape for the `window_on_all_layers` mutant (every cache
    /// bit intact, 1 152 hidden elements moved), and an earlier draft of
    /// `meets_bound` that checked only the caches passed it.
    #[test]
    fn a_defect_visible_only_in_the_final_hidden_still_fails_the_bound() {
        let g = geom();
        let a = filled(g, 1.0).with_final_hidden(vec![1.0, 2.0, 3.0]);
        let b = filled(g, 1.0).with_final_hidden(vec![1.0, 2.0, 3.25]);
        let v = compare(&a, &b).expect("compare");
        assert_eq!(v.max_abs_diff, 0.0, "the caches are untouched");
        assert!(v.per_position_max_abs.iter().all(|&d| d == 0.0));
        assert!(!v.bit_identical);
        let err = v
            .meets_bound(1.0e-3, 10.0)
            .expect_err("a last-layer-only defect must fail the bound");
        assert!(err.contains("final_hidden"), "{err}");
        // Same story for a bits-differ-at-zero-distance hidden change.
        let c = filled(g, 1.0).with_final_hidden(vec![1.0, 2.0, -0.0]);
        let d = filled(g, 1.0).with_final_hidden(vec![1.0, 2.0, 0.0]);
        let v2 = compare(&c, &d).expect("compare");
        assert_eq!(v2.hidden_max_abs_diff, 0.0);
        assert!(v2.meets_bound(1.0, 10.0).is_err());
    }

    #[test]
    fn geometry_mismatch_is_an_error_not_a_verdict() {
        let a = filled(geom(), 1.0);
        let mut g2 = geom();
        g2.positions = 7;
        let b = KvSnapshot::zeroed(g2);
        assert!(matches!(
            compare(&a, &b),
            Err(KvCompareError::GeometryMismatch { .. })
        ));
    }

    #[test]
    fn digest_distinguishes_signed_zero() {
        let g = geom();
        let mut a = KvSnapshot::zeroed(g);
        a.k[0][0] = 0.0;
        let mut b = KvSnapshot::zeroed(g);
        b.k[0][0] = -0.0;
        assert_ne!(a.digest(), b.digest());
        assert!(!compare(&a, &b).expect("compare").bit_identical);
    }

    #[test]
    fn rows_are_extracted_across_every_kv_head() {
        let g = geom();
        let a = filled(g, 1.0);
        let row = a.key_row(2, 3);
        assert_eq!(row.len(), g.n_kv_heads * g.head_dim);
        for h in 0..g.n_kv_heads {
            for d in 0..g.head_dim {
                assert_eq!(row[h * g.head_dim + d], a.k[2][g.index(h, 3, d)]);
            }
        }
    }
}
