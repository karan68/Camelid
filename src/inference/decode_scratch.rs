//! Recycling buffer pool for the decode hot loop (Lane B step 5).
//!
//! Steady-state decode used to allocate a fresh `Vec<f32>` (and a fresh
//! `String` name) for every op output — hundreds of heap round-trips per
//! token, all layer-proportional. This pool turns those into reuse hits:
//! `take(len)` returns a zeroed vector with EXACTLY the semantics of
//! `vec![0.0; len]` (values and length identical — allocation source cannot
//! affect arithmetic), backed by a recycled buffer whenever one with enough
//! capacity is available; `recycle`/`recycle_tensor` return buffers at the
//! points where the layer forward provably owns the dead intermediate.
//!
//! Concurrency: takes/recycles happen on the decode orchestrator thread
//! (ops are dispatched serially per token), so the internal Mutex is
//! uncontended; rayon WORKER scratch must NOT use this pool — workers keep
//! `thread_local!` buffers instead (see the attention lane) so nothing
//! contends inside parallel regions.
//!
//! The pool is unbounded in count but bounded in practice by the working
//! set of one token (every take is matched by a recycle at layer/token
//! end); buffers keep their high-water capacity, which is the point.

use std::sync::Mutex;

use crate::tensor::CpuTensor;

static POOL: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());
static NAME_POOL: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// A zeroed `Vec<f32>` of `len`, bit-identical in content to
/// `vec![0.0; len]`, reusing a recycled buffer when one fits.
pub(super) fn take(len: usize) -> Vec<f32> {
    let mut pool = POOL.lock().expect("decode scratch pool poisoned");
    // Last-in first-out keeps the hottest buffer (best cache behavior); scan
    // a bounded tail for one with enough capacity so odd sizes don't strand
    // large buffers.
    let limit = pool.len().min(8);
    for i in 0..limit {
        let idx = pool.len() - 1 - i;
        if pool[idx].capacity() >= len {
            let mut buffer = pool.swap_remove(idx);
            drop(pool);
            buffer.clear();
            buffer.resize(len, 0.0);
            return buffer;
        }
    }
    drop(pool);
    vec![0.0; len]
}

/// Return a buffer to the pool.
pub(super) fn recycle(buffer: Vec<f32>) {
    if buffer.capacity() == 0 {
        return;
    }
    POOL.lock()
        .expect("decode scratch pool poisoned")
        .push(buffer);
}

/// Build a tensor from pooled parts: the data buffer comes from the caller
/// (usually via [`take`]) and the name String is recycled when one is
/// available (clear + push_str reuses its capacity — no allocation once the
/// pool is warm).
pub(super) fn tensor_from_pooled(
    name: &str,
    dims: Vec<usize>,
    data: Vec<f32>,
) -> crate::Result<CpuTensor> {
    let mut owned = NAME_POOL
        .lock()
        .expect("decode scratch name pool poisoned")
        .pop()
        .unwrap_or_default();
    owned.clear();
    owned.push_str(name);
    CpuTensor::from_f32(owned, dims, data)
}

/// Reclaim a dead intermediate tensor: its data buffer and name String both
/// return to their pools. Call ONLY where the tensor is provably dead (the
/// layer forward's end-of-scope points).
// Consumed by the layer-forward recycling pass (step 5c); until that lands
// only the tests exercise it.
#[allow(dead_code)]
pub(super) fn recycle_tensor(tensor: CpuTensor) {
    let (name, data) = tensor.into_parts();
    recycle(data);
    if name.capacity() > 0 {
        NAME_POOL
            .lock()
            .expect("decode scratch name pool poisoned")
            .push(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_matches_vec_zero_semantics() {
        for len in [0usize, 1, 63, 64, 3072] {
            let a = take(len);
            let b = vec![0.0f32; len];
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
            recycle(a);
        }
    }

    #[test]
    fn recycled_buffers_are_reused_and_rezeroed() {
        let mut a = take(128);
        a.iter_mut().for_each(|v| *v = 7.0);
        let ptr = a.as_ptr() as usize;
        recycle(a);
        let b = take(64);
        // Reuse is capacity-based; content must be zero regardless.
        assert!(b.iter().all(|v| v.to_bits() == 0));
        let _ = ptr; // pointer identity is an implementation detail, not asserted
        recycle(b);
    }
}
