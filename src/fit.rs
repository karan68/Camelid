//! Model *fit* advisor — a capacity verdict for "can this machine run this model?".
//!
//! This is a **capacity axis only**. A [`FitVerdict::FitsResident`] means the
//! model's footprint fits the detected memory budget — it says **nothing** about
//! whether the model is *supported* (`COMPATIBILITY.md`) or *runnable-lane
//! anchored* (`crate::runnable`). Those are separate axes and must never be
//! conflated in copy or code.
//!
//! The math is a pure, GPU-free heuristic over byte counts, in the same spirit as
//! [`crate::cuda_vram::evaluate`] (which this module reuses for the VRAM branch).
//! It is **advisory**: the authoritative guards remain the mid-load
//! [`crate::cuda_vram::VramShortfall`] and the mid-generation
//! `BackendError::KvCacheBudgetExceeded` (`src/inference/kv_cache.rs`). This layer
//! only helps a user *choose* before they commit; it never gates a download and
//! never relaxes a runtime guard.
//!
//! On hosts where memory cannot be probed (e.g. macOS, where
//! [`crate::capability::HardwareProfile`] reports `host_ram_total_bytes == 0`) the
//! verdict degrades to [`FitVerdict::Unknown`] rather than guessing — an unknown
//! host must never read as "won't fit".

use crate::capability::HardwareProfile;

/// Share of *available* host RAM the advisor treats as usable, mirroring the
/// KV-cache budget policy in `src/inference/kv_cache.rs`
/// (`KV_CACHE_BUDGET_AVAILABLE_PERCENT`) so the advisor and the runtime guard
/// agree on what "usable RAM" means.
const USABLE_RAM_AVAILABLE_PERCENT: u64 = 80;
/// Floor as a share of *total* host RAM, mirroring
/// `KV_CACHE_BUDGET_TOTAL_FLOOR_PERCENT` — guards against a transient dip in the
/// live `available` reading (which drops sharply as weights fault into the
/// working set) collapsing the budget below what a normal run needs.
const USABLE_RAM_TOTAL_FLOOR_PERCENT: u64 = 25;

/// The advisor's verdict for a single (model footprint, host) pair.
///
/// Serialized in `snake_case` for the catalog API (Slice 2); the string form is
/// also exposed via [`FitVerdict::as_str`] for the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    /// Weights + KV fit within the GPU's VRAM (respecting headroom), or — with no
    /// usable GPU — within the host RAM budget. The comfortable case.
    FitsResident,
    /// Weights + KV exceed VRAM alone but fit the combined VRAM + host-RAM budget,
    /// i.e. the documented CUDA VRAM+host-RAM layer-offload split can carry it.
    FitsWithOffload,
    /// No usable GPU, but the footprint fits the host RAM budget (CPU backend).
    CpuOnlyOk,
    /// Exceeds every available budget on this host. The pick would fail at load or
    /// generation time; the UI should steer the user to a smaller/quantized row.
    WontFit,
    /// The host's memory could not be probed (e.g. macOS), so no honest capacity
    /// claim can be made. Advisory-blind: never treated as a failure.
    Unknown,
}

impl FitVerdict {
    /// Stable lowercase label, matching the serialized form. For CLI columns/logs.
    pub fn as_str(self) -> &'static str {
        match self {
            FitVerdict::FitsResident => "fits_resident",
            FitVerdict::FitsWithOffload => "fits_with_offload",
            FitVerdict::CpuOnlyOk => "cpu_only_ok",
            FitVerdict::WontFit => "wont_fit",
            FitVerdict::Unknown => "unknown",
        }
    }

    /// Whether the verdict says the model can run *somehow* on this host. `Unknown`
    /// is **not** runnable-negative — it is the absence of a claim — so it returns
    /// `false` here only in the sense of "no positive fit was proven". Callers that
    /// must not block on unknowns should test `!= WontFit` instead.
    pub fn is_positive_fit(self) -> bool {
        matches!(
            self,
            FitVerdict::FitsResident | FitVerdict::FitsWithOffload | FitVerdict::CpuOnlyOk
        )
    }

    /// Short human label for a CLI column or terse log. UI surfaces (WebUI) author
    /// their own copy; this is the terminal-facing wording.
    pub fn cli_label(self) -> &'static str {
        match self {
            FitVerdict::FitsResident => "fits",
            FitVerdict::FitsWithOffload => "fits (offload)",
            FitVerdict::CpuOnlyOk => "fits (CPU)",
            FitVerdict::WontFit => "too big",
            FitVerdict::Unknown => "unknown",
        }
    }
}

/// The footprint of a model to assess, in bytes.
///
/// `weight_bytes` is exact for curated catalog rows (`CatalogItem.size_bytes` is
/// the GGUF file size). `kv_bytes_at_ctx` is the projected key+value cache for the
/// context length being assessed; deriving it pre-download from architecture
/// metadata is the Slice-2 concern — this pure core simply takes both byte counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitInputs {
    pub weight_bytes: u64,
    pub kv_bytes_at_ctx: u64,
}

impl FitInputs {
    /// Total resident footprint (weights + KV), saturating.
    pub fn footprint_bytes(&self) -> u64 {
        self.weight_bytes.saturating_add(self.kv_bytes_at_ctx)
    }
}

/// The usable host-RAM budget in bytes, or `None` when RAM is unknown
/// (`host_ram_total_bytes == 0`). Mirrors the KV-cache budget policy:
/// `max(80% of available, 25% of total)`.
fn usable_host_ram_bytes(hw: &HardwareProfile) -> Option<u64> {
    if hw.host_ram_total_bytes == 0 {
        return None;
    }
    let by_available = hw
        .host_ram_free_bytes
        .saturating_mul(USABLE_RAM_AVAILABLE_PERCENT)
        / 100;
    let floor = hw
        .host_ram_total_bytes
        .saturating_mul(USABLE_RAM_TOTAL_FLOOR_PERCENT)
        / 100;
    Some(by_available.max(floor))
}

/// Whether the host has a GPU we can actually place weights on.
fn has_usable_gpu(hw: &HardwareProfile) -> bool {
    hw.cuda_available && hw.cuda_vram_free_bytes > 0
}

/// Pure fit decision with an explicit VRAM headroom (in MiB), so the whole thing
/// is deterministic and unit-testable without touching process env or a GPU.
///
/// Decision order (host-honest):
/// 1. Usable GPU present → try VRAM-resident via [`crate::cuda_vram::evaluate`].
///    - Ok → [`FitVerdict::FitsResident`].
///    - Shortfall → offload: fits VRAM + usable host RAM → [`FitVerdict::FitsWithOffload`];
///      RAM known but too small → [`FitVerdict::WontFit`]; RAM unknown → [`FitVerdict::Unknown`].
/// 2. No usable GPU → fits host RAM → [`FitVerdict::CpuOnlyOk`]; too small →
///    [`FitVerdict::WontFit`]; RAM unknown → [`FitVerdict::Unknown`].
fn assess_with_headroom(hw: &HardwareProfile, m: &FitInputs, vram_headroom_mib: u64) -> FitVerdict {
    let footprint = m.footprint_bytes();
    let usable_ram = usable_host_ram_bytes(hw);

    if has_usable_gpu(hw) {
        match crate::cuda_vram::evaluate(hw.cuda_vram_free_bytes, footprint, vram_headroom_mib) {
            Ok(_) => return FitVerdict::FitsResident,
            Err(_) => {
                return match usable_ram {
                    Some(ram) if footprint <= hw.cuda_vram_free_bytes.saturating_add(ram) => {
                        FitVerdict::FitsWithOffload
                    }
                    Some(_) => FitVerdict::WontFit,
                    None => FitVerdict::Unknown,
                };
            }
        }
    }

    match usable_ram {
        Some(ram) if footprint <= ram => FitVerdict::CpuOnlyOk,
        Some(_) => FitVerdict::WontFit,
        None => FitVerdict::Unknown,
    }
}

/// Assess whether `m` fits `hw`, using the configured VRAM headroom
/// ([`crate::cuda_vram::min_headroom_mib`], env `CAMELID_MIN_VRAM_HEADROOM_MIB`).
///
/// This is the public entry point. It is deterministic given the process env and
/// the passed hardware profile; the pure arithmetic lives in
/// [`assess_with_headroom`] for env-free testing.
pub fn assess(hw: &HardwareProfile, m: &FitInputs) -> FitVerdict {
    assess_with_headroom(hw, m, crate::cuda_vram::min_headroom_mib())
}

/// Advisory allowance, as a percent of weight bytes, for everything resident
/// beyond the weights at a modest default context: the KV cache, activations, and
/// scratch. This is a deliberately coarse, deliberately *conservative* (slightly
/// over-estimating) heuristic for the **pre-download** badge — the exact KV cost
/// is architecture- and context-specific and is enforced at runtime by the KV
/// predict-and-abort guard (`src/inference/kv_cache.rs`). Over-estimating keeps a
/// "fits" badge safe rather than optimistic. A per-architecture bound is a future
/// refinement; a flat pad avoids inventing per-model dimensions we cannot know
/// before the GGUF is on disk.
pub const ADVISORY_OVERHEAD_PERCENT: u64 = 25;

/// Build [`FitInputs`] for a catalog row from its known weight footprint
/// (`CatalogItem.size_bytes`), padding by [`ADVISORY_OVERHEAD_PERCENT`] to stand
/// in for KV + activations at a modest context. The pad is carried in
/// `kv_bytes_at_ctx`; it is an estimate, not a measured KV size.
pub fn advisory_footprint(weight_bytes: u64) -> FitInputs {
    let overhead = weight_bytes.saturating_mul(ADVISORY_OVERHEAD_PERCENT) / 100;
    FitInputs {
        weight_bytes,
        kv_bytes_at_ctx: overhead,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::SimdCaps;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    /// A hardware profile with only the memory-relevant fields set; everything
    /// else defaulted. Keeps the fit tests focused on the capacity math.
    fn profile(
        cuda_available: bool,
        vram_free_bytes: u64,
        ram_total_bytes: u64,
        ram_free_bytes: u64,
    ) -> HardwareProfile {
        HardwareProfile {
            cuda_available,
            cuda_device_count: if cuda_available { 1 } else { 0 },
            cuda_device_name: None,
            cuda_compute_capability: None,
            cuda_tensor_cores: false,
            cuda_vram_total_bytes: vram_free_bytes,
            cuda_vram_free_bytes: vram_free_bytes,
            cpu_logical_cores: 8,
            host_ram_total_bytes: ram_total_bytes,
            host_ram_free_bytes: ram_free_bytes,
            simd: SimdCaps::default(),
        }
    }

    fn inputs(weight_bytes: u64, kv_bytes: u64) -> FitInputs {
        FitInputs {
            weight_bytes,
            kv_bytes_at_ctx: kv_bytes,
        }
    }

    // A small headroom so tests reason in round GiB without the default 512 MiB
    // nudging boundary cases.
    const H: u64 = 0;

    #[test]
    fn resident_when_footprint_fits_vram_with_headroom() {
        // 8 GB card, a ~3.4 GB weight + 0.5 GB KV = ~3.9 GB → resident.
        let hw = profile(true, 8 * GIB, 16 * GIB, 12 * GIB);
        let m = inputs(3_421_898_816, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, 512), FitVerdict::FitsResident);
    }

    #[test]
    fn headroom_pushes_a_tight_fit_out_of_resident() {
        // Footprint is just under free VRAM, but the 512 MiB headroom is violated,
        // so it must NOT be resident. With host RAM available it becomes offload.
        let hw = profile(true, 8 * GIB, 32 * GIB, 24 * GIB);
        let m = inputs(8 * GIB - 100 * MIB, 0);
        let verdict = assess_with_headroom(&hw, &m, 512);
        assert_eq!(verdict, FitVerdict::FitsWithOffload);
    }

    #[test]
    fn offload_when_weights_exceed_vram_but_fit_vram_plus_ram() {
        // 8B Q8_0 (~8.5 GB) on a 6 GB card with 32 GB RAM → VRAM+host-RAM offload.
        let hw = profile(true, 6 * GIB, 32 * GIB, 24 * GIB);
        let m = inputs(8_541_283_552, 512 * MIB);
        assert_eq!(
            assess_with_headroom(&hw, &m, H),
            FitVerdict::FitsWithOffload
        );
    }

    #[test]
    fn wont_fit_when_footprint_exceeds_vram_plus_ram() {
        // Tiny VRAM + tiny RAM cannot carry a 12 GB model even with offload.
        let hw = profile(true, 2 * GIB, 4 * GIB, 3 * GIB);
        let m = inputs(12 * GIB, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::WontFit);
    }

    #[test]
    fn cpu_only_ok_when_no_gpu_and_fits_ram() {
        // No GPU, 16 GB RAM (healthy) → 80%-of-available = ~9.6 GB budget carries a
        // ~3.4 GB model comfortably.
        let hw = profile(false, 0, 16 * GIB, 12 * GIB);
        let m = inputs(3_421_898_816, 256 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::CpuOnlyOk);
    }

    #[test]
    fn wont_fit_cpu_only_when_model_exceeds_ram_budget() {
        // No GPU, 8 GB RAM → budget ~ max(80% of 5 GB=4 GB, 25% of 8 GB=2 GB)=4 GB;
        // an 8.5 GB model won't fit.
        let hw = profile(false, 0, 8 * GIB, 5 * GIB);
        let m = inputs(8_541_283_552, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::WontFit);
    }

    #[test]
    fn ram_floor_dominates_when_available_dips_transiently() {
        // 32 GB total but a transient low available (2 GB). The 25%-of-total floor
        // (8 GB) must dominate the 80%-of-available (1.6 GB), so a ~3.4 GB model
        // still fits CPU-only rather than spuriously "won't fit".
        let hw = profile(false, 0, 32 * GIB, 2 * GIB);
        let m = inputs(3_421_898_816, 256 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::CpuOnlyOk);
    }

    #[test]
    fn unknown_when_ram_unprobed_and_no_gpu() {
        // macOS-style: RAM probe returns 0 and no CUDA. No honest claim possible.
        let hw = profile(false, 0, 0, 0);
        let m = inputs(3_421_898_816, 256 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::Unknown);
    }

    #[test]
    fn unknown_when_gpu_overflows_and_ram_unprobed() {
        // GPU present but too small, and RAM cannot be probed → offload can't be
        // judged → Unknown (never WontFit on an unknown host).
        let hw = profile(true, 2 * GIB, 0, 0);
        let m = inputs(8_541_283_552, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::Unknown);
    }

    #[test]
    fn cuda_flag_without_vram_is_not_a_usable_gpu() {
        // cuda_available=true but 0 free VRAM → treated as CPU host; fits RAM.
        let hw = profile(true, 0, 16 * GIB, 12 * GIB);
        let m = inputs(2 * GIB, 128 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::CpuOnlyOk);
    }

    #[test]
    fn footprint_saturates_and_wont_fit_on_extreme_values() {
        let m = inputs(u64::MAX, u64::MAX);
        assert_eq!(m.footprint_bytes(), u64::MAX);
        let hw = profile(false, 0, 16 * GIB, 12 * GIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::WontFit);
    }

    #[test]
    fn verdict_labels_are_stable() {
        assert_eq!(FitVerdict::FitsResident.as_str(), "fits_resident");
        assert_eq!(FitVerdict::FitsWithOffload.as_str(), "fits_with_offload");
        assert_eq!(FitVerdict::CpuOnlyOk.as_str(), "cpu_only_ok");
        assert_eq!(FitVerdict::WontFit.as_str(), "wont_fit");
        assert_eq!(FitVerdict::Unknown.as_str(), "unknown");
        assert!(FitVerdict::FitsResident.is_positive_fit());
        assert!(FitVerdict::FitsWithOffload.is_positive_fit());
        assert!(FitVerdict::CpuOnlyOk.is_positive_fit());
        assert!(!FitVerdict::WontFit.is_positive_fit());
        assert!(!FitVerdict::Unknown.is_positive_fit());
    }

    #[test]
    fn verdict_serializes_to_snake_case() {
        let json = serde_json::to_string(&FitVerdict::FitsWithOffload).unwrap();
        assert_eq!(json, "\"fits_with_offload\"");
    }
}
