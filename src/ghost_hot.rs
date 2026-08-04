//! Sparse local shadow files for Ghost-MoE models.
//!
//! A Ghost-MoE run reads routed experts from its `.cghost` artifact, but the
//! tokenizer, tied embedding/head, attention weights, shared FFN, router and
//! norms still use the source GGUF. When that source lives on a slow external
//! drive, those always-hot pages remain on the token critical path.
//!
//! [`write_moe_hot_shadow`] creates a byte-address-compatible GGUF on the fast
//! drive. It copies the whole source except the two routed-expert tensors in
//! every layer. Those ranges are left as sparse holes apart from tiny,
//! deterministic identity islands sampled from every expert. All original
//! tensor offsets and the logical file length remain unchanged while physical
//! storage contains the common core plus the bounded pairing proof. The shadow
//! is intentionally useful only with the matching `.cghost`; reading an expert
//! directly from it still yields mostly zeroes and invalid weights.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::Path;

use crate::gguf::GgufFile;
use crate::ghost::{ensure_distinct_artifact_paths, moe_identity_sample_range};
use crate::platform_fs::read_exact_at;
use crate::{BackendError, Result};

const COPY_CHUNK_BYTES: usize = 1024 * 1024;
const GATE_UP_EXPERT_SUFFIX: &str = ".ffn_gate_up_exps.weight";
const DOWN_EXPERT_SUFFIX: &str = ".ffn_down_exps.weight";

/// Storage report for a completed sparse hot-shadow conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostHotShadowReport {
    /// Logical file length. This is identical to the source GGUF length.
    pub logical_bytes: u64,
    /// Filesystem blocks allocated after `sync_all` (best effort on Windows).
    pub allocated_bytes: u64,
    /// Source bytes physically copied into the shadow.
    pub copied_bytes: u64,
    /// Routed-expert bytes deliberately left as sparse zero holes.
    pub sparse_expert_bytes: u64,
    /// Number of skipped tensor extents (two per transformer layer).
    pub expert_extents: usize,
}

struct ExpertSpanPlan {
    expert_spans: Vec<Range<u64>>,
    identity_islands: Vec<Range<u64>>,
}

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::InvalidModelMetadata(message.into())
}

fn io_err(path: &Path, source: std::io::Error) -> BackendError {
    BackendError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn expert_tensor_layer(name: &str, suffix: &str) -> Option<usize> {
    let stem = name.strip_suffix(suffix)?;
    let layer = stem.strip_prefix("blk.")?;
    if layer.is_empty() || (layer.len() > 1 && layer.starts_with('0')) {
        return None;
    }
    layer.parse().ok()
}

fn validate_and_collect_expert_spans(
    gguf: &GgufFile,
    layer_range: Option<Range<usize>>,
    source_len: u64,
) -> Result<ExpertSpanPlan> {
    if gguf.architecture() != Some("gemma4") {
        return Err(invalid(format!(
            "Ghost hot shadow requires general.architecture=gemma4, found {:?}",
            gguf.architecture()
        )));
    }
    let block_count = gguf
        .metadata_u32("gemma4.block_count")
        .ok_or_else(|| invalid("Ghost hot shadow requires gemma4.block_count"))?
        as usize;
    let expert_count = gguf
        .metadata_u32("gemma4.expert_count")
        .ok_or_else(|| invalid("Ghost hot shadow requires gemma4.expert_count"))?
        as usize;
    if block_count == 0 || expert_count == 0 {
        return Err(invalid(format!(
            "Ghost hot shadow requires non-zero layer/expert counts, found layers={block_count} experts={expert_count}"
        )));
    }
    let requested = layer_range.unwrap_or(0..block_count);
    if requested != (0..block_count) {
        return Err(invalid(format!(
            "Ghost hot shadow must cover the full all-layer MoE model 0..{block_count}; partial range {requested:?} would expose missing expert bytes as weights"
        )));
    }

    let mut found = vec![[false; 2]; block_count];
    let mut spans = Vec::with_capacity(block_count * 2);
    let mut identity_islands = Vec::with_capacity(block_count * expert_count * 2);
    for tensor in &gguf.tensors {
        let role = if tensor.name.ends_with(GATE_UP_EXPERT_SUFFIX) {
            Some((GATE_UP_EXPERT_SUFFIX, 0usize))
        } else if tensor.name.ends_with(DOWN_EXPERT_SUFFIX) {
            Some((DOWN_EXPERT_SUFFIX, 1usize))
        } else {
            None
        };
        let Some((suffix, role_idx)) = role else {
            continue;
        };
        let layer = expert_tensor_layer(&tensor.name, suffix).ok_or_else(|| {
            invalid(format!(
                "routed-expert tensor {} does not use canonical blk.N naming",
                tensor.name
            ))
        })?;
        if layer >= block_count {
            return Err(invalid(format!(
                "routed-expert tensor {} names layer {layer}, outside block_count {block_count}",
                tensor.name
            )));
        }
        if std::mem::replace(&mut found[layer][role_idx], true) {
            return Err(invalid(format!(
                "duplicate routed-expert tensor role {} for layer {layer}",
                if role_idx == 0 { "gate_up" } else { "down" }
            )));
        }
        if tensor.dimensions.last().copied() != Some(expert_count as u64) {
            return Err(invalid(format!(
                "routed-expert tensor {} last dimension {:?} does not match expert_count {expert_count}",
                tensor.name,
                tensor.dimensions.last()
            )));
        }
        if !tensor.n_bytes.is_multiple_of(expert_count as u64) {
            return Err(invalid(format!(
                "routed-expert tensor {} byte length {} is not divisible by expert_count {expert_count}",
                tensor.name, tensor.n_bytes
            )));
        }
        let end = tensor
            .absolute_offset
            .checked_add(tensor.n_bytes)
            .ok_or_else(|| invalid(format!("tensor {} byte range overflows", tensor.name)))?;
        if end > source_len {
            return Err(invalid(format!(
                "tensor {} range {}..{end} exceeds source length {source_len}",
                tensor.name, tensor.absolute_offset
            )));
        }
        spans.push(tensor.absolute_offset..end);
        let expert_len = tensor.n_bytes / expert_count as u64;
        let role = if role_idx == 0 {
            "gate_up_exps"
        } else {
            "down_exps"
        };
        for expert_idx in 0..expert_count {
            let slice_start = tensor.absolute_offset + expert_idx as u64 * expert_len;
            identity_islands.push(moe_identity_sample_range(
                slice_start,
                expert_len,
                layer,
                expert_idx,
                role,
            ));
        }
    }

    for (layer, roles) in found.iter().enumerate() {
        if !roles[0] || !roles[1] {
            return Err(invalid(format!(
                "Ghost hot shadow requires gate_up and down routed-expert tensors in every layer; layer {layer} has gate_up={} down={}",
                roles[0], roles[1]
            )));
        }
    }
    spans.sort_unstable_by_key(|span| span.start);
    for pair in spans.windows(2) {
        if pair[0].end > pair[1].start {
            return Err(invalid(format!(
                "routed-expert tensor ranges overlap: {:?} and {:?}",
                pair[0], pair[1]
            )));
        }
    }
    identity_islands.sort_unstable_by_key(|span| span.start);
    for pair in identity_islands.windows(2) {
        if pair[0].end > pair[1].start {
            return Err(invalid(format!(
                "Ghost hot-shadow identity islands overlap: {:?} and {:?}",
                pair[0], pair[1]
            )));
        }
    }
    Ok(ExpertSpanPlan {
        expert_spans: spans,
        identity_islands,
    })
}

fn sparse_hole_spans(
    expert_spans: &[Range<u64>],
    identity_islands: &[Range<u64>],
) -> Result<Vec<Range<u64>>> {
    let mut holes = Vec::with_capacity(expert_spans.len() + identity_islands.len());
    let mut island_idx = 0usize;
    for expert in expert_spans {
        while island_idx < identity_islands.len()
            && identity_islands[island_idx].end <= expert.start
        {
            island_idx += 1;
        }
        let mut cursor = expert.start;
        while island_idx < identity_islands.len() && identity_islands[island_idx].start < expert.end
        {
            let island = &identity_islands[island_idx];
            if island.start < expert.start || island.end > expert.end {
                return Err(invalid(format!(
                    "Ghost hot-shadow identity island {island:?} is not contained in expert extent {expert:?}"
                )));
            }
            if cursor < island.start {
                holes.push(cursor..island.start);
            }
            cursor = island.end;
            island_idx += 1;
        }
        if cursor < expert.end {
            holes.push(cursor..expert.end);
        }
    }
    if island_idx != identity_islands.len() {
        return Err(invalid(format!(
            "Ghost hot-shadow identity island {:?} is outside every expert extent",
            identity_islands[island_idx]
        )));
    }
    Ok(holes)
}

fn copy_span(
    source: &File,
    destination: &mut File,
    span: Range<u64>,
    scratch: &mut [u8],
    source_path: &Path,
    destination_path: &Path,
) -> Result<u64> {
    if span.start >= span.end {
        return Ok(0);
    }
    destination
        .seek(SeekFrom::Start(span.start))
        .map_err(|err| io_err(destination_path, err))?;
    let mut cursor = span.start;
    while cursor < span.end {
        let take = (span.end - cursor).min(scratch.len() as u64) as usize;
        read_exact_at(source, &mut scratch[..take], cursor)
            .map_err(|err| io_err(source_path, err))?;
        destination
            .write_all(&scratch[..take])
            .map_err(|err| io_err(destination_path, err))?;
        cursor += take as u64;
    }
    Ok(span.end - span.start)
}

#[cfg(unix)]
fn allocated_bytes(metadata: &std::fs::Metadata, _path: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.blocks().saturating_mul(512)
}

#[cfg(windows)]
fn allocated_bytes(metadata: &std::fs::Metadata, path: &Path) -> u64 {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{GetLastError, SetLastError, ERROR_SUCCESS};
    use windows_sys::Win32::Storage::FileSystem::{GetCompressedFileSizeW, INVALID_FILE_SIZE};

    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let mut high = 0u32;
    // SAFETY: `wide` is a live, NUL-terminated UTF-16 path and `high` points to
    // writable storage. GetCompressedFileSize reports allocated bytes for sparse
    // files on NTFS/ReFS. Fall back conservatively when the filesystem declines.
    unsafe { SetLastError(ERROR_SUCCESS) };
    let low = unsafe { GetCompressedFileSizeW(wide.as_ptr(), &mut high) };
    if low == INVALID_FILE_SIZE && unsafe { GetLastError() } != ERROR_SUCCESS {
        metadata.len()
    } else {
        (u64::from(high) << 32) | u64::from(low)
    }
}

#[cfg(not(any(unix, windows)))]
fn allocated_bytes(metadata: &std::fs::Metadata, _path: &Path) -> u64 {
    metadata.len()
}

#[cfg(windows)]
fn prepare_sparse_destination(destination: &File, destination_path: &Path) -> Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::Ioctl::FSCTL_SET_SPARSE;
    use windows_sys::Win32::System::IO::DeviceIoControl;

    let mut returned = 0u32;
    // SAFETY: destination owns a live writable file handle. A null input buffer
    // with FSCTL_SET_SPARSE marks the file sparse; no output buffer is required.
    let ok = unsafe {
        DeviceIoControl(
            destination.as_raw_handle(),
            FSCTL_SET_SPARSE,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &mut returned,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(io_err(destination_path, std::io::Error::last_os_error()));
    }
    Ok(())
}

#[cfg(not(windows))]
fn prepare_sparse_destination(_destination: &File, _destination_path: &Path) -> Result<()> {
    Ok(())
}

/// APFS eagerly accounts zero-extended ranges as allocated blocks. Explicitly
/// punch the page-aligned interior of every requested hole after the retained
/// bytes have been copied; partial edge blocks stay allocated but remain zero.
#[cfg(target_os = "macos")]
fn punch_expert_holes(
    destination: &File,
    spans: &[Range<u64>],
    destination_path: &Path,
) -> Result<()> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;

    let metadata = destination
        .metadata()
        .map_err(|err| io_err(destination_path, err))?;
    let block = metadata.blksize().max(1);
    for span in spans {
        let start = span.start.div_ceil(block) * block;
        let end = (span.end / block) * block;
        if start >= end {
            continue;
        }
        let mut hole = libc::fpunchhole_t {
            fp_flags: 0,
            reserved: 0,
            fp_offset: start as libc::off_t,
            fp_length: (end - start) as libc::off_t,
        };
        // SAFETY: fd is a live writable regular file; `hole` is initialized,
        // block-aligned and wholly inside the file.
        let rc = unsafe { libc::fcntl(destination.as_raw_fd(), libc::F_PUNCHHOLE, &mut hole) };
        if rc == -1 {
            return Err(io_err(destination_path, std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

/// Linux's explicit sparse-hole operation. Like the APFS path, retain partial
/// edge blocks and deallocate every whole filesystem block inside each tensor.
#[cfg(target_os = "linux")]
fn punch_expert_holes(
    destination: &File,
    spans: &[Range<u64>],
    destination_path: &Path,
) -> Result<()> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;

    let metadata = destination
        .metadata()
        .map_err(|err| io_err(destination_path, err))?;
    let block = metadata.blksize().max(1);
    for span in spans {
        let start = span.start.div_ceil(block) * block;
        let end = (span.end / block) * block;
        if start >= end {
            continue;
        }
        // SAFETY: fd is a live writable regular file and the aligned range is
        // wholly inside it. KEEP_SIZE preserves GGUF's logical length.
        let rc = unsafe {
            libc::fallocate(
                destination.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                start as libc::off_t,
                (end - start) as libc::off_t,
            )
        };
        if rc == -1 {
            return Err(io_err(destination_path, std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

/// Windows sparse ranges are deallocated with FSCTL_SET_ZERO_DATA. The file is
/// marked sparse before its logical length is extended, so this operation keeps
/// only the copied common core and the bounded identity islands allocated.
#[cfg(windows)]
fn punch_expert_holes(
    destination: &File,
    spans: &[Range<u64>],
    destination_path: &Path,
) -> Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::Ioctl::{FILE_ZERO_DATA_INFORMATION, FSCTL_SET_ZERO_DATA};
    use windows_sys::Win32::System::IO::DeviceIoControl;

    for span in spans {
        if span.start >= span.end {
            continue;
        }
        let info = FILE_ZERO_DATA_INFORMATION {
            FileOffset: i64::try_from(span.start)
                .map_err(|_| invalid("Ghost hot-shadow hole start exceeds Windows i64 offsets"))?,
            BeyondFinalZero: i64::try_from(span.end)
                .map_err(|_| invalid("Ghost hot-shadow hole end exceeds Windows i64 offsets"))?,
        };
        let mut returned = 0u32;
        // SAFETY: destination owns a live writable sparse-file handle and info
        // describes an in-file half-open byte range. The control code consumes
        // exactly one FILE_ZERO_DATA_INFORMATION and returns no output payload.
        let ok = unsafe {
            DeviceIoControl(
                destination.as_raw_handle(),
                FSCTL_SET_ZERO_DATA,
                std::ptr::from_ref(&info).cast(),
                std::mem::size_of::<FILE_ZERO_DATA_INFORMATION>() as u32,
                std::ptr::null_mut(),
                0,
                &mut returned,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io_err(destination_path, std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
fn punch_expert_holes(
    _destination: &File,
    _spans: &[Range<u64>],
    _destination_path: &Path,
) -> Result<()> {
    // Other targets retain the correct zero-hole byte semantics. Their platform
    // APIs are not exposed by this crate, so allocation reporting is conservative.
    Ok(())
}

/// Create an offset-compatible sparse GGUF containing only a Gemma 4 MoE
/// model's always-hot common core.
///
/// The source metadata must describe a full all-layer Gemma 4 MoE model with
/// one fused `ffn_gate_up_exps` and one `ffn_down_exps` tensor per layer.
/// A 128-byte deterministic identity island is retained per expert slice and
/// role so a matching fingerprinted `.cghost` can validate the pair quickly.
/// `layer_range` exists to share the repacker CLI's range option, but every
/// value except `None`/the full `0..block_count` range is refused. The source
/// and destination must be different files.
pub fn write_moe_hot_shadow(
    source_path: &Path,
    destination_path: &Path,
    gguf: &GgufFile,
    layer_range: Option<Range<usize>>,
) -> Result<GhostHotShadowReport> {
    ensure_distinct_artifact_paths(&[
        ("source GGUF", source_path),
        ("hot-shadow output", destination_path),
    ])?;

    let source = File::open(source_path).map_err(|err| io_err(source_path, err))?;
    let source_len = source
        .metadata()
        .map_err(|err| io_err(source_path, err))?
        .len();
    let ExpertSpanPlan {
        expert_spans,
        identity_islands,
    } = validate_and_collect_expert_spans(gguf, layer_range, source_len)?;
    let hole_spans = sparse_hole_spans(&expert_spans, &identity_islands)?;

    // File::set_len creates an unwritten sparse address space on APFS/ext4.
    // Only the complement of the expert extents is subsequently written.
    let mut destination = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(destination_path)
        .map_err(|err| io_err(destination_path, err))?;
    prepare_sparse_destination(&destination, destination_path)?;
    destination
        .set_len(source_len)
        .map_err(|err| io_err(destination_path, err))?;

    let mut scratch = vec![0u8; COPY_CHUNK_BYTES];
    let mut cursor = 0u64;
    let mut copied_bytes = 0u64;
    let expert_bytes: u64 = expert_spans.iter().map(|span| span.end - span.start).sum();
    for span in &expert_spans {
        copied_bytes += copy_span(
            &source,
            &mut destination,
            cursor..span.start,
            &mut scratch,
            source_path,
            destination_path,
        )?;
        cursor = span.end;
    }
    copied_bytes += copy_span(
        &source,
        &mut destination,
        cursor..source_len,
        &mut scratch,
        source_path,
        destination_path,
    )?;
    for island in &identity_islands {
        copied_bytes += copy_span(
            &source,
            &mut destination,
            island.clone(),
            &mut scratch,
            source_path,
            destination_path,
        )?;
    }
    let identity_bytes: u64 = identity_islands
        .iter()
        .map(|span| span.end - span.start)
        .sum();
    let sparse_expert_bytes = expert_bytes
        .checked_sub(identity_bytes)
        .ok_or_else(|| invalid("Ghost hot-shadow identity bytes exceed routed-expert bytes"))?;
    punch_expert_holes(&destination, &hole_spans, destination_path)?;
    destination
        .sync_all()
        .map_err(|err| io_err(destination_path, err))?;
    let metadata = destination
        .metadata()
        .map_err(|err| io_err(destination_path, err))?;
    debug_assert_eq!(metadata.len(), source_len);
    debug_assert_eq!(copied_bytes + sparse_expert_bytes, source_len);

    Ok(GhostHotShadowReport {
        logical_bytes: source_len,
        allocated_bytes: allocated_bytes(&metadata, destination_path),
        copied_bytes,
        sparse_expert_bytes,
        expert_extents: expert_spans.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{read_metadata, GgufMetadataValue};
    use std::io::Read;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i32(bytes: &mut Vec<u8>, value: i32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i64(bytes: &mut Vec<u8>, value: i64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_string(bytes, key);
        push_u32(bytes, 8); // GGUF_TYPE_STRING
        push_string(bytes, value);
    }

    fn push_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        push_u32(bytes, 4); // GGUF_TYPE_UINT32
        push_u32(bytes, value);
    }

    struct TensorFixture<'a> {
        name: &'a str,
        dims: &'a [i64],
        dtype: i32,
        payload: Vec<u8>,
    }

    fn write_moe_fixture(path: &Path) {
        // Large expert tensors ensure the filesystem allocation check can
        // distinguish written bytes from holes even with coarse block sizes.
        let expert_len = 32usize * 4096 * 2 / 32 * 18;
        let tensors = vec![
            TensorFixture {
                name: "token_embd.weight",
                dims: &[8],
                dtype: 0,
                payload: vec![0x11; 32],
            },
            TensorFixture {
                name: "blk.0.ffn_gate_up_exps.weight",
                dims: &[32, 4096, 2],
                dtype: 2,
                payload: vec![0x22; expert_len],
            },
            TensorFixture {
                name: "blk.0.ffn_gate_inp.weight",
                dims: &[8],
                dtype: 0,
                payload: vec![0x33; 32],
            },
            TensorFixture {
                name: "blk.0.ffn_down_exps.weight",
                dims: &[32, 4096, 2],
                dtype: 2,
                payload: vec![0x44; expert_len],
            },
            TensorFixture {
                name: "output_norm.weight",
                dims: &[8],
                dtype: 0,
                payload: vec![0x55; 32],
            },
        ];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_i64(&mut bytes, tensors.len() as i64);
        push_i64(&mut bytes, 6);
        push_kv_string(&mut bytes, "general.architecture", "gemma4");
        push_kv_u32(&mut bytes, "general.alignment", 32);
        push_kv_u32(&mut bytes, "gemma4.block_count", 1);
        push_kv_u32(&mut bytes, "gemma4.expert_count", 2);
        push_kv_u32(&mut bytes, "gemma4.expert_used_count", 1);
        push_kv_string(&mut bytes, "general.name", "hot-shadow-fixture");

        let mut relative_offset = 0u64;
        for tensor in &tensors {
            push_string(&mut bytes, tensor.name);
            push_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                push_i64(&mut bytes, *dim);
            }
            push_i32(&mut bytes, tensor.dtype);
            push_u64(&mut bytes, relative_offset);
            relative_offset = (relative_offset + tensor.payload.len() as u64).next_multiple_of(32);
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        for tensor in tensors {
            bytes.extend_from_slice(&tensor.payload);
            while !bytes.len().is_multiple_of(32) {
                bytes.push(0);
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn hot_shadow_preserves_metadata_common_bytes_and_sparse_expert_holes() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.gguf");
        let shadow_path = dir.path().join("shadow.gguf");
        write_moe_fixture(&source_path);
        let source_gguf = read_metadata(&source_path).unwrap();

        let report = write_moe_hot_shadow(&source_path, &shadow_path, &source_gguf, None)
            .expect("write sparse hot shadow");
        let shadow_gguf = read_metadata(&shadow_path).expect("shadow remains parseable GGUF");
        assert_eq!(shadow_gguf.version, source_gguf.version);
        assert_eq!(shadow_gguf.tensor_count, source_gguf.tensor_count);
        assert_eq!(shadow_gguf.metadata, source_gguf.metadata);
        assert_eq!(shadow_gguf.tensors, source_gguf.tensors);

        let source = std::fs::read(&source_path).unwrap();
        let shadow = std::fs::read(&shadow_path).unwrap();
        assert_eq!(shadow.len(), source.len());
        assert_eq!(report.logical_bytes, source.len() as u64);
        assert_eq!(report.expert_extents, 2);
        assert_eq!(
            report.copied_bytes + report.sparse_expert_bytes,
            report.logical_bytes
        );
        assert_eq!(
            &shadow[..source_gguf.data_start_offset as usize],
            &source[..source_gguf.data_start_offset as usize],
            "header, metadata and tensor directory are byte-exact"
        );

        for tensor in &source_gguf.tensors {
            let range =
                tensor.absolute_offset as usize..(tensor.absolute_offset + tensor.n_bytes) as usize;
            if tensor.name.ends_with(GATE_UP_EXPERT_SUFFIX)
                || tensor.name.ends_with(DOWN_EXPERT_SUFFIX)
            {
                assert!(source[range.clone()].iter().any(|&byte| byte != 0));
                let (suffix, role) = if tensor.name.ends_with(GATE_UP_EXPERT_SUFFIX) {
                    (GATE_UP_EXPERT_SUFFIX, "gate_up_exps")
                } else {
                    (DOWN_EXPERT_SUFFIX, "down_exps")
                };
                let layer = expert_tensor_layer(&tensor.name, suffix).unwrap();
                let expert_count = 2usize;
                let expert_len = tensor.n_bytes / expert_count as u64;
                let mut retained = vec![false; tensor.n_bytes as usize];
                for expert_idx in 0..expert_count {
                    let slice_start = tensor.absolute_offset + expert_idx as u64 * expert_len;
                    let island =
                        moe_identity_sample_range(slice_start, expert_len, layer, expert_idx, role);
                    for offset in island.start..island.end {
                        retained[(offset - tensor.absolute_offset) as usize] = true;
                    }
                }
                for (offset, (&actual, &original)) in shadow[range.clone()]
                    .iter()
                    .zip(&source[range.clone()])
                    .enumerate()
                {
                    if retained[offset] {
                        assert_eq!(actual, original, "identity island byte {offset}");
                    } else {
                        assert_eq!(actual, 0, "expert hole byte {offset}");
                    }
                }
                assert!(shadow[range].iter().any(|&byte| byte != 0));
            } else {
                assert_eq!(&shadow[range.clone()], &source[range]);
            }
        }

        #[cfg(unix)]
        assert!(
            report.allocated_bytes < report.logical_bytes,
            "expert ranges should consume holes, allocated={} logical={}",
            report.allocated_bytes,
            report.logical_bytes
        );
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_SPARSE_FILE;

            let attributes = std::fs::metadata(&shadow_path).unwrap().file_attributes();
            assert_ne!(
                attributes & FILE_ATTRIBUTE_SPARSE_FILE,
                0,
                "Windows hot shadow must be explicitly marked sparse"
            );
            assert!(report.allocated_bytes <= report.logical_bytes);
        }
    }

    #[test]
    fn hot_shadow_refuses_partial_or_missing_layer_experts_before_truncating_output() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.gguf");
        let shadow_path = dir.path().join("shadow.gguf");
        write_moe_fixture(&source_path);
        let gguf = read_metadata(&source_path).unwrap();
        let err = write_moe_hot_shadow(&source_path, &shadow_path, &gguf, Some(0..0)).unwrap_err();
        assert!(err.to_string().contains("full all-layer"));
        assert!(!shadow_path.exists());

        let mut missing = gguf.clone();
        missing
            .metadata
            .insert("gemma4.block_count".into(), GgufMetadataValue::U32(2));
        let err = write_moe_hot_shadow(&source_path, &shadow_path, &missing, None).unwrap_err();
        assert!(err.to_string().contains("layer 1"));
        assert!(!shadow_path.exists());
    }

    #[test]
    fn hot_shadow_refuses_source_as_destination() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.gguf");
        write_moe_fixture(&source_path);
        let gguf = read_metadata(&source_path).unwrap();
        let before_len = std::fs::metadata(&source_path).unwrap().len();
        let err = write_moe_hot_shadow(&source_path, &source_path, &gguf, None).unwrap_err();
        assert!(err.to_string().contains("distinct files"));
        assert_eq!(std::fs::metadata(&source_path).unwrap().len(), before_len);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn hot_shadow_refuses_hard_link_to_source_before_truncating() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.gguf");
        let shadow_path = dir.path().join("shadow.gguf");
        write_moe_fixture(&source_path);
        let gguf = read_metadata(&source_path).unwrap();
        let before = std::fs::read(&source_path).unwrap();
        std::fs::hard_link(&source_path, &shadow_path).unwrap();

        let err = write_moe_hot_shadow(&source_path, &shadow_path, &gguf, None).unwrap_err();
        assert!(err.to_string().contains("distinct files"));
        assert_eq!(std::fs::read(&source_path).unwrap(), before);
        assert_eq!(std::fs::read(&shadow_path).unwrap(), before);
    }

    #[test]
    fn fixture_expert_payloads_are_readable_before_shadowing() {
        // Keeps `Read` imported and pins that fixture holes replace real bytes,
        // not already-zero padding.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.gguf");
        write_moe_fixture(&path);
        let gguf = read_metadata(&path).unwrap();
        let expert = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name.ends_with(GATE_UP_EXPERT_SUFFIX))
            .unwrap();
        let mut file = File::open(path).unwrap();
        file.seek(SeekFrom::Start(expert.absolute_offset)).unwrap();
        let mut probe = [0u8; 16];
        file.read_exact(&mut probe).unwrap();
        assert_eq!(probe, [0x22; 16]);
    }
}
