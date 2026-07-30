//! STAMPEDE Phase 1 — Windows core topology and optional worker pinning for
//! the dedicated decode/prefill rayon pools.
//!
//! **Hybrid decode sizing (default on, hybrid hosts only).** Intel hybrid parts
//! expose performance and efficiency cores that the pre-hybrid
//! `GetLogicalProcessorInformation` cannot tell apart, so the raw physical-core
//! count it returns silently mixes both classes and the decode pool is sized
//! for cores that decode ~2.4x slower. `GetLogicalProcessorInformationEx`
//! reports a per-core `EfficiencyClass`; when a host reports more than one
//! class, [`hybrid_decode_core_count`] returns the number of top-class cores
//! and the decode pool is sized to that instead. Hosts reporting a single class
//! (pre-hybrid Intel, AMD, VMs, servers) get `None` and keep their existing
//! width exactly. `CAMELID_WIN_HYBRID_POLICY=0|false|off|no` restores it.
//!
//! This changes a thread *count*, not the arithmetic, so it is bit-identical by
//! construction — and it deliberately does not pin. Measured on an i9-14900HX,
//! n=6 paired server starts per arm (one process per sample, since the pool is
//! built once per process): sizing the decode pool to the performance-core
//! count is worth +5.30 tok/s over the shipped default, 95% CI [+4.99, +5.60],
//! 6/6, for 55% less CPU time on a byte-identical answer.
//!
//! Pinning that pool is not merely unnecessary on top of the width, it costs.
//! Against the same pool left unpinned, a hard affinity mask scores -1.69 tok/s
//! (CI [-2.27, -1.11], 0/6) and an ideal-processor hint -0.55 (CI [-1.15,
//! +0.06]). Once the pool stops oversubscribing, the scheduler keeps the busy
//! threads on performance cores by itself, and a mask only stops it recovering
//! when something else wants the core.
//!
//! The width wants to be the performance-core count specifically, not merely
//! something below the physical count: sweeping the same lever peaks there and
//! falls off on both sides — against the shipped default, 4 workers score
//! -2.98 tok/s, 8 score +4.92, 12 +3.66 and 16 +1.96 (n=6 paired, every
//! comparison significant).
//!
//! **`CAMELID_WIN_PIN` (default off, unchanged).** The pre-existing opt-in that
//! spreads pool workers over *every* physical core as `worker % cores`:
//! * `ideal` — `SetThreadIdealProcessor` placement hint (soft; the scheduler
//!   may still migrate under pressure).
//! * `hard`  — `SetThreadAffinityMask` to the worker's physical core (both
//!   SMT siblings stay in the mask, so the core is owned but the scheduler
//!   can still bounce between its siblings).
//!
//! Every path fails open: detection failure or an empty selection leaves the
//! pre-change width and no pinning.

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
use std::sync::OnceLock;

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WinPinMode {
    Off,
    Ideal,
    Hard,
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub(super) fn win_pin_mode() -> WinPinMode {
    static MODE: OnceLock<WinPinMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("CAMELID_WIN_PIN") {
            Ok(value) => {
                let value = value.trim();
                if value.eq_ignore_ascii_case("ideal") {
                    WinPinMode::Ideal
                } else if value.eq_ignore_ascii_case("hard") {
                    WinPinMode::Hard
                } else {
                    // Unknown values (and explicit off/0) stay unpinned: the
                    // pinned lanes are opt-in and fail-open.
                    WinPinMode::Off
                }
            }
            Err(_) => WinPinMode::Off,
        }
    })
}

/// Masks of the top-`EfficiencyClass` cores, or `None` when this host is not
/// hybrid and the policy must not engage.
///
/// Kept pure and target-independent so the "single efficiency class changes
/// nothing" guarantee is unit-testable without Windows hybrid hardware.
/// `process_affinity` is intersected in so a restricted process (job object,
/// `start /affinity`) is never sized for cores it cannot run on.
#[cfg_attr(
    not(all(target_os = "windows", target_arch = "x86_64")),
    allow(dead_code)
)]
fn performance_core_masks(
    masks: &[usize],
    efficiency_classes: &[u8],
    process_affinity: usize,
) -> Option<Vec<usize>> {
    if masks.is_empty() || masks.len() != efficiency_classes.len() {
        return None;
    }
    let top = *efficiency_classes.iter().max()?;
    if efficiency_classes.iter().all(|class| *class == top) {
        return None;
    }
    let selected: Vec<usize> = masks
        .iter()
        .zip(efficiency_classes.iter())
        .filter(|(_, class)| **class == top)
        .map(|(mask, _)| *mask & process_affinity)
        .filter(|mask| *mask != 0)
        .collect();
    (!selected.is_empty()).then_some(selected)
}

/// `false` only for an explicit `CAMELID_WIN_HYBRID_POLICY=0|false|off|no`.
#[cfg_attr(
    not(all(target_os = "windows", target_arch = "x86_64")),
    allow(dead_code)
)]
fn hybrid_policy_enabled() -> bool {
    hybrid_policy_enabled_from(std::env::var("CAMELID_WIN_HYBRID_POLICY").ok().as_deref())
}

/// Split from the environment read so the gate is testable without an env race.
#[cfg_attr(
    not(all(target_os = "windows", target_arch = "x86_64")),
    allow(dead_code)
)]
fn hybrid_policy_enabled_from(value: Option<&str>) -> bool {
    match value {
        Some(value) => {
            let value = value.trim();
            !(value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("no"))
        }
        None => true,
    }
}

/// `GROUP_AFFINITY` (winnt.h). Mirrored locally because the record walk needs
/// a stable layout over a variable-length buffer.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
#[repr(C)]
struct GroupAffinity {
    mask: usize,
    group: u16,
    reserved: [u16; 3],
}

/// `PROCESSOR_RELATIONSHIP` (winnt.h).
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
#[repr(C)]
struct ProcessorRelationship {
    flags: u8,
    efficiency_class: u8,
    reserved: [u8; 20],
    group_count: u16,
    group_mask: [GroupAffinity; 1],
}

/// `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` narrowed to the processor-core arm.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
#[repr(C)]
struct LogicalProcessorInformationEx {
    relationship: i32,
    size: u32,
    processor: ProcessorRelationship,
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const _: () = {
    assert!(std::mem::size_of::<GroupAffinity>() == 16);
    assert!(std::mem::size_of::<ProcessorRelationship>() == 40);
    assert!(std::mem::size_of::<LogicalProcessorInformationEx>() == 48);
};

/// Per-physical-core sibling masks plus efficiency classes, in OS enumeration
/// order. Group 0 only, which bounds the sizing decision to one group's cores.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn detect_core_topology() -> Option<(Vec<usize>, Vec<u8>)> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    const RELATION_PROCESSOR_CORE: i32 = 0;
    unsafe {
        let mut len: u32 = 0;
        // First call sizes the buffer (fails with ERROR_INSUFFICIENT_BUFFER).
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len);
        if len == 0 {
            return None;
        }
        // u64 backing keeps the buffer 8-aligned for the record walk.
        let mut buf: Vec<u64> = vec![0; (len as usize).div_ceil(8)];
        let base = buf.as_mut_ptr().cast::<u8>();
        if GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            base.cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(),
            &mut len,
        ) == 0
        {
            return None;
        }
        let total = len as usize;
        let record_size = std::mem::size_of::<LogicalProcessorInformationEx>();
        let mut masks = Vec::new();
        let mut classes = Vec::new();
        let mut offset = 0usize;
        // Records are variable length; `size` is the only way to advance.
        while offset + record_size <= total {
            let record = base.add(offset).cast::<LogicalProcessorInformationEx>();
            let size = (*record).size as usize;
            if size < record_size || offset + size > total {
                break;
            }
            if (*record).relationship == RELATION_PROCESSOR_CORE {
                let processor = &(*record).processor;
                if processor.group_count >= 1 {
                    let affinity = &processor.group_mask[0];
                    if affinity.group == 0 && affinity.mask != 0 {
                        masks.push(affinity.mask);
                        classes.push(processor.efficiency_class);
                    }
                }
            }
            offset += size;
        }
        (!masks.is_empty()).then_some((masks, classes))
    }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn process_affinity_mask() -> usize {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessAffinityMask};
    let mut process: usize = 0;
    let mut system: usize = 0;
    // SAFETY: both out-params are owned locals; the pseudo-handle needs no close.
    let ok = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) };
    if ok == 0 || process == 0 {
        usize::MAX
    } else {
        process
    }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn core_topology() -> Option<&'static (Vec<usize>, Vec<u8>)> {
    static TOPOLOGY: OnceLock<Option<(Vec<usize>, Vec<u8>)>> = OnceLock::new();
    TOPOLOGY.get_or_init(detect_core_topology).as_ref()
}

/// Per-physical-core logical-processor masks from
/// `GetLogicalProcessorInformation` (one `RelationProcessorCore` record per
/// core, its `ProcessorMask` covering that core's SMT siblings). `None` when
/// detection fails; order matches the OS enumeration order of cores.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn windows_core_masks() -> Option<&'static Vec<usize>> {
    static MASKS: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    MASKS
        .get_or_init(|| {
            use windows_sys::Win32::System::SystemInformation::{
                GetLogicalProcessorInformation, SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
            };
            const RELATION_PROCESSOR_CORE: i32 = 0;
            unsafe {
                let mut len: u32 = 0;
                // First call sizes the buffer (fails with ERROR_INSUFFICIENT_BUFFER).
                GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len);
                if len == 0 {
                    return None;
                }
                let count =
                    len as usize / std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
                if count == 0 {
                    return None;
                }
                let mut buf: Vec<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> = Vec::with_capacity(count);
                if GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) == 0 {
                    return None;
                }
                buf.set_len(count);
                let masks: Vec<usize> = buf
                    .iter()
                    .filter(|info| info.Relationship == RELATION_PROCESSOR_CORE)
                    .map(|info| info.ProcessorMask)
                    .filter(|mask| *mask != 0)
                    .collect();
                (!masks.is_empty()).then_some(masks)
            }
        })
        .as_ref()
}

/// Top-efficiency-class core masks for this host, or `None` when the policy is
/// disabled, the host is not hybrid, or detection failed.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn hybrid_performance_masks() -> Option<&'static Vec<usize>> {
    static MASKS: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    MASKS
        .get_or_init(|| {
            if !hybrid_policy_enabled() {
                return None;
            }
            let (masks, classes) = core_topology()?;
            let selected = performance_core_masks(masks, classes, process_affinity_mask());
            if let Some(selected) = selected.as_ref() {
                tracing::info!(
                    performance_cores = selected.len(),
                    physical_cores = masks.len(),
                    "hybrid CPU detected: sizing the decode pool to the performance cores"
                );
            }
            selected
        })
        .as_ref()
}

/// Decode-pool width for a hybrid host: one worker per performance core, or
/// `None` when the host is not hybrid so callers keep their existing sizing.
/// Only defined where the caller is, which is why there is no stub arm.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub(super) fn hybrid_decode_core_count() -> Option<usize> {
    hybrid_performance_masks().map(Vec::len)
}

/// Pin the calling pool worker (index `worker`) per the selected mode.
/// Shared by the decode and prefill pool `start_handler`s.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub(super) fn pin_pool_worker(pool: &'static str, worker: usize) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadAffinityMask, SetThreadIdealProcessor,
    };
    let mode = win_pin_mode();
    if mode == WinPinMode::Off {
        return;
    }
    let Some(masks) = windows_core_masks() else {
        return;
    };
    let core = worker % masks.len();
    let mask = masks[core];
    // SAFETY: GetCurrentThread returns a pseudo-handle that needs no
    // CloseHandle; both Set* calls only reconfigure the calling thread's
    // scheduling and cannot alias memory.
    unsafe {
        match mode {
            WinPinMode::Hard => {
                if SetThreadAffinityMask(GetCurrentThread(), mask) == 0 {
                    tracing::debug!(pool, worker, core, mask, "SetThreadAffinityMask failed");
                }
            }
            WinPinMode::Ideal => {
                let ideal = mask.trailing_zeros();
                if SetThreadIdealProcessor(GetCurrentThread(), ideal) == u32::MAX {
                    tracing::debug!(pool, worker, ideal, "SetThreadIdealProcessor failed");
                }
            }
            WinPinMode::Off => unreachable!(),
        }
    }
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            ?mode,
            cores = masks.len(),
            "CAMELID_WIN_PIN: pinning pool workers to physical cores"
        );
    });
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
pub(super) fn pin_pool_worker(_pool: &'static str, _worker: usize) {}

#[cfg(test)]
mod tests {
    use super::{hybrid_policy_enabled_from, performance_core_masks};

    const UNRESTRICTED: usize = usize::MAX;

    #[test]
    fn a_single_efficiency_class_disables_the_policy() {
        // Pre-hybrid Intel, every AMD part, VMs and servers report one class.
        // They must keep their existing decode pool width exactly.
        let masks = vec![0b0011, 0b1100, 0b0011_0000, 0b1100_0000];
        let classes = vec![0, 0, 0, 0];
        assert_eq!(performance_core_masks(&masks, &classes, UNRESTRICTED), None);
    }

    #[test]
    fn hybrid_topology_selects_only_the_top_class() {
        // Shape of an i9-14900HX: 8 SMT performance cores, then 16 single-thread
        // efficiency cores.
        let mut masks = Vec::new();
        let mut classes = Vec::new();
        for core in 0..8 {
            masks.push(0b11usize << (core * 2));
            classes.push(1);
        }
        for core in 0..16 {
            masks.push(1usize << (16 + core));
            classes.push(0);
        }
        let selected = performance_core_masks(&masks, &classes, UNRESTRICTED).expect("hybrid host");
        assert_eq!(selected.len(), 8);
        assert_eq!(selected.iter().fold(0usize, |acc, mask| acc | mask), 0xFFFF);
    }

    #[test]
    fn more_than_two_classes_keeps_only_the_fastest() {
        let masks = vec![0b0001, 0b0010, 0b0100];
        let classes = vec![2, 1, 0];
        assert_eq!(
            performance_core_masks(&masks, &classes, UNRESTRICTED),
            Some(vec![0b0001])
        );
    }

    #[test]
    fn process_affinity_narrows_the_selection() {
        let masks = vec![0b0011, 0b1100, 0b0001_0000];
        let classes = vec![1, 1, 0];
        assert_eq!(
            performance_core_masks(&masks, &classes, 0b0011),
            Some(vec![0b0011])
        );
    }

    #[test]
    fn an_affinity_without_a_performance_core_disables_the_policy() {
        let masks = vec![0b0011, 0b1100, 0b0001_0000];
        let classes = vec![1, 1, 0];
        assert_eq!(performance_core_masks(&masks, &classes, 0b0001_0000), None);
    }

    #[test]
    fn a_malformed_topology_is_rejected() {
        assert_eq!(performance_core_masks(&[], &[], UNRESTRICTED), None);
        assert_eq!(performance_core_masks(&[0b1], &[0, 1], UNRESTRICTED), None);
    }

    #[test]
    fn the_hybrid_policy_is_on_unless_explicitly_disabled() {
        assert!(hybrid_policy_enabled_from(None));
        assert!(hybrid_policy_enabled_from(Some("1")));
        assert!(hybrid_policy_enabled_from(Some("")));
        // An unrecognised value must not silently disable a default-on policy.
        assert!(hybrid_policy_enabled_from(Some("maybe")));
        for disabled in ["0", "false", "off", "no", "OFF", " off "] {
            assert!(
                !hybrid_policy_enabled_from(Some(disabled)),
                "{disabled} should disable the policy"
            );
        }
    }
}
