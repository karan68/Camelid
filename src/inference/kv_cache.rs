use std::env;

use serde::Serialize;

use crate::{
    model::{DenseLlamaDims, LlamaModelConfig},
    BackendError, Result,
};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaKvCachePlan {
    pub max_sequence_length: usize,
    pub layer_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub key_shape: Vec<usize>,
    pub value_shape: Vec<usize>,
}

impl LlamaKvCachePlan {
    pub fn from_config(config: &LlamaModelConfig) -> Result<Self> {
        let dims = DenseLlamaDims::from_config(config)?;
        let max_sequence_length = config.context_length as usize;
        let shape = vec![
            dims.block_count,
            max_sequence_length,
            dims.attention_head_count_kv,
            dims.head_dim,
        ];
        Ok(Self {
            max_sequence_length,
            layer_count: dims.block_count,
            kv_head_count: dims.attention_head_count_kv,
            head_dim: dims.head_dim,
            key_shape: shape.clone(),
            value_shape: shape,
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaKvCacheTrace {
    pub layer_index: usize,
    pub position_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub key_value_width: usize,
    pub key_checksum: f64,
    pub value_checksum: f64,
    pub key_rms: f32,
    pub value_rms: f32,
    pub key_max_abs: f32,
    pub key_max_abs_position: usize,
    pub key_max_abs_index: usize,
    pub value_max_abs: f32,
    pub value_max_abs_position: usize,
    pub value_max_abs_index: usize,
    pub sampled_positions: Vec<LlamaKvCachePositionTrace>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaKvCachePositionTrace {
    pub position: usize,
    pub key_checksum: f64,
    pub value_checksum: f64,
    pub key_rms: f32,
    pub value_rms: f32,
    pub key_max_abs: f32,
    pub value_max_abs: f32,
    pub key_first_values: Vec<f32>,
    pub value_first_values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LlamaKvCache {
    pub plan: LlamaKvCachePlan,
    /// Physical K/V arrangement; see [`KvLayout`]. Fixed at construction.
    pub layout: KvLayout,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub allocated_sequence_length: usize,
    pub position: usize,
    /// Max f32 K+V bytes this session may materialize before the predict-and-abort
    /// guard in `ensure_position_capacity` refuses. Host-derived operational config
    /// (env / available RAM), NOT cache state — excluded from `PartialEq`. `pub(super)`
    /// so sibling session code can carry it onto a hollow placeholder cache without
    /// re-resolving (which would re-query host RAM every step).
    pub(super) kv_budget_bytes: u64,
}

impl PartialEq for LlamaKvCache {
    fn eq(&self, other: &Self) -> bool {
        // Cache STATE only. `kv_budget_bytes` is host-derived operational config and
        // must not affect equality (the session PartialEq compares caches across runs).
        self.plan == other.plan
            && self.layout == other.layout
            && self.keys == other.keys
            && self.values == other.values
            && self.allocated_sequence_length == other.allocated_sequence_length
            && self.position == other.position
    }
}

/// Physical arrangement of the K/V buffers. Chosen ONCE at cache
/// construction (`BACKENDINFERENCE_KV_LAYOUT_HEAD_MAJOR`, default off) and
/// carried on the cache; every element address goes through the layout-aware
/// accessors below, so the choice never appears in hot loops as more than a
/// resolved stride. Values and arithmetic are identical in both layouts —
/// only addresses differ — so the head-major lane carries a bitwise-identity
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// `[position, layer, kv_head, head_dim]` — the historical layout; one
    /// token's row is contiguous, one head's positions sit a full token
    /// stride apart.
    PositionMajor,
    /// `[layer, kv_head, position, head_dim]` — each head's K/V is one
    /// contiguous stream over positions (stride = `allocated_sequence_length
    /// * head_dim`, re-laid-out on growth); decode reads become sequential.
    HeadMajor,
}

/// Env gate for the head-major layout, read once per cache construction.
/// Windows-first per the standing directive; the mechanism is arch-agnostic
/// and lifting the gate is a one-line decision.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn kv_layout_head_major_enabled() -> bool {
    super::q8_runtime::q8_0_env_flag_enabled_default_off("BACKENDINFERENCE_KV_LAYOUT_HEAD_MAJOR")
}

impl LlamaKvCache {
    pub fn new(plan: LlamaKvCachePlan) -> Result<Self> {
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        let layout = if kv_layout_head_major_enabled() {
            KvLayout::HeadMajor
        } else {
            KvLayout::PositionMajor
        };
        #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
        let layout = KvLayout::PositionMajor;
        Self::new_with_layout(plan, layout)
    }

    /// Explicit-layout constructor so the bitwise-identity tests can build
    /// both layouts without touching process env.
    pub fn new_with_layout(plan: LlamaKvCachePlan, layout: KvLayout) -> Result<Self> {
        Ok(Self {
            plan,
            layout,
            keys: Vec::new(),
            values: Vec::new(),
            allocated_sequence_length: 0,
            position: 0,
            kv_budget_bytes: resolve_kv_cache_budget_bytes(),
        })
    }

    pub fn can_append(&self) -> bool {
        self.position < self.plan.max_sequence_length
    }

    /// Roll the cache back to an earlier `position`, discarding newer
    /// entries. In both layouts `position` alone bounds what attention
    /// reads, so entries past the rollback point are dead until overwritten
    /// by later appends — no buffer work is needed. Used by speculative
    /// decoding to drop rejected draft tokens.
    pub fn rollback_to_position(&mut self, position: usize) -> Result<()> {
        if position > self.position {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "KV rollback target {position} is beyond current position {}",
                self.position
            )));
        }
        self.position = position;
        Ok(())
    }

    pub(super) fn ensure_position_capacity(
        &mut self,
        required_sequence_length: usize,
    ) -> Result<()> {
        if required_sequence_length > self.plan.max_sequence_length {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "KV cache position {required_sequence_length} exceeds context length {}",
                self.plan.max_sequence_length
            )));
        }
        if required_sequence_length <= self.allocated_sequence_length {
            return Ok(());
        }
        let target_sequence_length = self.grow_sequence_length(required_sequence_length);
        // Predict-and-abort on host memory (conductor §9): the f32 K+V cache is the
        // dominant, otherwise-uncapped allocation. Project the bytes of the ACTUAL
        // (post-rounding) growth and refuse BEFORE the `resize` below — the host ceiling
        // must never be discovered by OOMing mid-generation. Budget is
        // `CAMELID_MAX_KV_CACHE_BYTES`, else a fraction of available physical RAM
        // (Windows); unbounded where neither is known.
        let projected_bytes =
            (target_sequence_length as u64).saturating_mul(self.kv_bytes_per_token());
        if projected_bytes > self.kv_budget_bytes {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "KV cache growth to {target_sequence_length} positions needs {projected_bytes} \
                 bytes of f32 K+V, above the {} byte budget for this host; reduce the prompt/context \
                 length or set {KV_CACHE_BUDGET_LIMIT_ENV} deliberately for a controlled run",
                self.kv_budget_bytes
            )));
        }
        let values = target_sequence_length
            .checked_mul(self.plan.layer_count)
            .and_then(|value| value.checked_mul(self.plan.kv_head_count))
            .and_then(|value| value.checked_mul(self.plan.head_dim))
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch("KV cache element count overflow".to_string())
            })?;
        match self.layout {
            KvLayout::PositionMajor => {
                // Positions are outermost, so growth is a pure append.
                self.keys.resize(values, 0.0);
                self.values.resize(values, 0.0);
            }
            KvLayout::HeadMajor => {
                // Each head's stream stride is the allocated capacity, so
                // growth re-lays the buffer: copy every (layer, head) block
                // to its new base. Pure permutation of identical bits —
                // bitwise-neutral — amortized over the growth chunking.
                let head_dim = self.plan.head_dim;
                let old_len = self.allocated_sequence_length * head_dim;
                let streams = self.plan.layer_count * self.plan.kv_head_count;
                let new_len = target_sequence_length * head_dim;
                let mut new_keys = vec![0.0f32; values];
                let mut new_values = vec![0.0f32; values];
                if old_len > 0 {
                    for stream in 0..streams {
                        let old_base = stream * old_len;
                        let new_base = stream * new_len;
                        new_keys[new_base..new_base + old_len]
                            .copy_from_slice(&self.keys[old_base..old_base + old_len]);
                        new_values[new_base..new_base + old_len]
                            .copy_from_slice(&self.values[old_base..old_base + old_len]);
                    }
                }
                self.keys = new_keys;
                self.values = new_values;
            }
        }
        self.allocated_sequence_length = target_sequence_length;
        Ok(())
    }

    fn grow_sequence_length(&self, required_sequence_length: usize) -> usize {
        let grow_tokens = kv_cache_grow_tokens(self.plan.max_sequence_length);
        if grow_tokens <= 1 {
            return required_sequence_length;
        }
        required_sequence_length
            .div_ceil(grow_tokens)
            .saturating_mul(grow_tokens)
            .min(self.plan.max_sequence_length)
    }

    pub fn allocated_elements(&self) -> usize {
        self.keys.len() + self.values.len()
    }

    pub fn allocated_bytes(&self) -> u64 {
        (self.allocated_elements() as u64) * (std::mem::size_of::<f32>() as u64)
    }

    pub(super) fn offset(&self, layer_idx: usize, position: usize, kv_head: usize) -> usize {
        match self.layout {
            KvLayout::PositionMajor => {
                (((position * self.plan.layer_count) + layer_idx) * self.plan.kv_head_count
                    + kv_head)
                    * self.plan.head_dim
            }
            KvLayout::HeadMajor => {
                (((layer_idx * self.plan.kv_head_count) + kv_head)
                    * self.allocated_sequence_length
                    + position)
                    * self.plan.head_dim
            }
        }
    }

    pub(super) fn head_base_offset(&self, layer_idx: usize, kv_head: usize) -> usize {
        self.offset(layer_idx, 0, kv_head)
    }

    /// Element step between one head's consecutive positions — the stride the
    /// per-head attention walks take. A full token stride in position-major,
    /// `head_dim` (contiguous) in head-major.
    pub(super) fn head_position_stride(&self) -> usize {
        match self.layout {
            KvLayout::PositionMajor => self.position_stride(),
            KvLayout::HeadMajor => self.plan.head_dim,
        }
    }

    /// Elements ONE TOKEN occupies per buffer across all layers/heads —
    /// layout-independent count used for capacity and budget math, NOT an
    /// address stride (use [`Self::head_position_stride`] for walks).
    pub(super) fn position_stride(&self) -> usize {
        self.plan.layer_count * self.plan.kv_head_count * self.plan.head_dim
    }

    /// f32 bytes one token's KV occupies across all layers/heads, counting both the
    /// K and V buffers — the per-token cost the predict-and-abort guard projects.
    fn kv_bytes_per_token(&self) -> u64 {
        (self.position_stride() as u64)
            .saturating_mul(2) // K + V
            .saturating_mul(std::mem::size_of::<f32>() as u64)
    }
}

fn kv_cache_grow_tokens(max_sequence_length: usize) -> usize {
    if max_sequence_length < 512 {
        return 1;
    }
    env::var("CAMELID_KV_CACHE_GROW_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(256)
}

/// Explicit override for the KV-cache predict-and-abort budget, in bytes. Mirrors
/// `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES` for the weight-materialization guard.
const KV_CACHE_BUDGET_LIMIT_ENV: &str = "CAMELID_MAX_KV_CACHE_BYTES";
/// Default share of *available* physical RAM the KV cache may claim when no env override
/// is set — leaves headroom for activations + the OS so a long-context request fails
/// closed instead of pushing the host into paging / OOM.
const KV_CACHE_BUDGET_AVAILABLE_PERCENT: u64 = 80;

/// Resolve a new session's KV-cache memory budget (bytes): an explicit
/// `CAMELID_MAX_KV_CACHE_BYTES` wins (deterministic, for tuning or controlled runs);
/// otherwise a fraction of available physical RAM (Windows `GlobalMemoryStatusEx`).
fn resolve_kv_cache_budget_bytes() -> u64 {
    let env_value = env::var(KV_CACHE_BUDGET_LIMIT_ENV).ok();
    kv_cache_budget_from(env_value.as_deref(), crate::gait::host_ram_status())
}

/// Pure budget policy (extracted so it is testable without mutating process env or
/// querying the host): a parseable non-empty env override wins; else a fraction of
/// available RAM; else (no env, no RAM probe — e.g. off Windows) unbounded.
fn kv_cache_budget_from(env_value: Option<&str>, ram: Option<(u64, u64)>) -> u64 {
    if let Some(trimmed) = env_value.map(str::trim) {
        if !trimmed.is_empty() {
            if let Ok(bytes) = trimmed.parse::<u64>() {
                return bytes;
            }
        }
    }
    match ram {
        Some((_total, available)) => {
            available.saturating_mul(KV_CACHE_BUDGET_AVAILABLE_PERCENT) / 100
        }
        None => u64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_with(
        max_seq: usize,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> LlamaKvCachePlan {
        let shape = vec![layers, max_seq, kv_heads, head_dim];
        LlamaKvCachePlan {
            max_sequence_length: max_seq,
            layer_count: layers,
            kv_head_count: kv_heads,
            head_dim,
            key_shape: shape.clone(),
            value_shape: shape,
        }
    }

    #[test]
    fn kv_bytes_per_token_counts_k_and_v_f32() {
        // Llama 3.2 3B shape: 28 layers * 8 kv-heads * 128 head_dim = 28672 stride;
        // *2 (K+V) *4 (f32) = 229376 bytes/token.
        let cache = LlamaKvCache::new(plan_with(131072, 28, 8, 128)).unwrap();
        assert_eq!(cache.kv_bytes_per_token(), 229_376);
        // TinyLlama shape: 22 * 4 * 64 = 5632 stride; *8 = 45056 bytes/token.
        let cache = LlamaKvCache::new(plan_with(2048, 22, 4, 64)).unwrap();
        assert_eq!(cache.kv_bytes_per_token(), 45_056);
    }

    #[test]
    fn predict_and_abort_refuses_over_budget_before_allocating() {
        // max_seq < 512 => grow_tokens == 1, so target == required (no chunk rounding) —
        // deterministic regardless of CAMELID_KV_CACHE_GROW_TOKENS.
        let mut cache = LlamaKvCache::new(plan_with(400, 16, 8, 64)).unwrap();
        let per_token = cache.kv_bytes_per_token(); // 16*8*64*8 = 65536
        cache.kv_budget_bytes = 100 * per_token; // budget for exactly 100 tokens
                                                 // At budget: allowed, allocates exactly 100.
        assert!(cache.ensure_position_capacity(100).is_ok());
        assert_eq!(cache.allocated_sequence_length, 100);
        // One token over: refused BEFORE any new allocation.
        let err = cache.ensure_position_capacity(101).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("budget"),
            "message should explain the budget: {msg}"
        );
        assert!(
            msg.contains(KV_CACHE_BUDGET_LIMIT_ENV),
            "message should name the override env: {msg}"
        );
        assert_eq!(
            cache.allocated_sequence_length, 100,
            "the over-budget length must not have been allocated"
        );
    }

    #[test]
    fn unbounded_budget_allows_normal_growth() {
        let mut cache = LlamaKvCache::new(plan_with(4096, 16, 8, 64)).unwrap();
        cache.kv_budget_bytes = u64::MAX;
        assert!(cache.ensure_position_capacity(2048).is_ok());
        assert!(cache.allocated_sequence_length >= 2048);
    }

    #[test]
    fn budget_policy_env_override_wins_then_ram_fraction_then_unbounded() {
        // Explicit env override (parseable) wins verbatim.
        assert_eq!(
            kv_cache_budget_from(Some(" 4096 "), Some((32 << 30, 16 << 30))),
            4096
        );
        // Unparseable / empty env falls through to the RAM fraction.
        assert_eq!(
            kv_cache_budget_from(Some("not-a-number"), Some((32 << 30, 10 << 30))),
            (10u64 << 30) * KV_CACHE_BUDGET_AVAILABLE_PERCENT / 100
        );
        assert_eq!(
            kv_cache_budget_from(None, Some((32 << 30, 10 << 30))),
            (10u64 << 30) * KV_CACHE_BUDGET_AVAILABLE_PERCENT / 100
        );
        // No env and no RAM probe (e.g. off Windows) -> unbounded (env remains the gate).
        assert_eq!(kv_cache_budget_from(None, None), u64::MAX);
    }

    /// On macOS the host RAM probe now returns `Some`, so the no-env path resolves to
    /// the 80%-of-available RAM branch instead of the off-platform `u64::MAX` unbounded
    /// fallback — the auto-budget genuinely gates here now. (The per-token byte math and
    /// refuse-before-allocate tests above are platform-independent and already cover the
    /// Mac guard path; this pins the policy resolution that changed for Mac.)
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_ram_branch_engages_auto_budget() {
        // resolve_kv_cache_budget_bytes() below does a libc getenv; serialize it against every
        // other env-mutating test (ENV_LOCK) so the read can't race a concurrent set_var on the
        // shared `environ` under parallel `cargo test`. (See the review's phase1 F1/F2.)
        let _env = crate::test_support::env_lock();
        let ram = crate::gait::host_ram_status();
        assert!(ram.is_some(), "macOS host_ram_status must report Some");
        let budget = kv_cache_budget_from(None, ram);
        assert!(budget > 0, "auto-budget must be a positive byte count");
        assert!(
            budget < u64::MAX,
            "auto-budget must be bounded on Mac, not the off-platform unbounded fallback"
        );
        if let Some((total, available)) = ram {
            assert!(available <= total, "available must not exceed total");
            assert_eq!(
                budget,
                available.saturating_mul(KV_CACHE_BUDGET_AVAILABLE_PERCENT) / 100
            );
        }
        // An explicit override still wins verbatim over the live RAM reading.
        assert_eq!(kv_cache_budget_from(Some("4096"), ram), 4096);
        // And the live end-to-end resolver agrees (no env set in this test).
        assert!(resolve_kv_cache_budget_bytes() < u64::MAX);
    }

    #[test]
    fn budget_excluded_from_state_equality() {
        let mut a = LlamaKvCache::new(plan_with(2048, 22, 4, 64)).unwrap();
        let mut b = LlamaKvCache::new(plan_with(2048, 22, 4, 64)).unwrap();
        a.kv_budget_bytes = 1 << 20;
        b.kv_budget_bytes = 1 << 40;
        // Different host budgets, identical state -> still equal.
        assert_eq!(a, b);
    }
}
