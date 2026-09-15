//! Trusted process-local load hashes. Disk cache entries never authorize a model.
//! Unix change time catches same-size rewrites even when mtime is restored. On
//! filesystems without this identity, hash again. Verification stays uncached.
use super::{hex_lower, ReceiptError};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Debug, Serialize)]
pub struct HashProgress {
    pub filename: String,
    pub bytes_read: u64,
    pub total_bytes: u64,
}

#[derive(Clone, PartialEq, Eq)]
struct Identity {
    len: u64,
    modified: std::time::SystemTime,
    device: u64,
    inode: u64,
    changed: (i64, i64),
}

#[cfg(unix)]
fn identity(meta: &std::fs::Metadata) -> Option<Identity> {
    use std::os::unix::fs::MetadataExt;
    Some(Identity {
        len: meta.len(),
        modified: meta.modified().ok()?,
        device: meta.dev(),
        inode: meta.ino(),
        changed: (meta.ctime(), meta.ctime_nsec()),
    })
}
#[cfg(not(unix))]
fn identity(_: &std::fs::Metadata) -> Option<Identity> {
    None
}

#[derive(Default)]
struct LoadHashes {
    completed: Mutex<BTreeMap<PathBuf, (Identity, String)>>,
    active: Mutex<BTreeMap<PathBuf, HashProgress>>,
    // Serialize cold reads to avoid competing multi-GB USB reads and duplicate
    // work. Health only takes `active`, never this lock.
    reading: Mutex<()>,
}
static HASHES: OnceLock<LoadHashes> = OnceLock::new();

pub fn progress() -> Vec<HashProgress> {
    HASHES
        .get_or_init(LoadHashes::default)
        .active
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .cloned()
        .collect()
}

pub fn sha256_file_hex_for_load(path: &Path) -> Result<String, ReceiptError> {
    HASHES.get_or_init(LoadHashes::default).hash(path)
}

impl LoadHashes {
    fn hash(&self, path: &Path) -> Result<String, ReceiptError> {
        let io_err = |source| ReceiptError::Io {
            path: path.to_path_buf(),
            source,
        };
        let canonical = std::fs::canonicalize(path).map_err(io_err)?;
        let _reading = self.reading.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = std::fs::File::open(&canonical).map_err(io_err)?;
        let metadata = file.metadata().map_err(io_err)?;
        let before = identity(&metadata);
        let started = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if let Some(key) = before.as_ref() {
            if let Some((old, digest)) = self
                .completed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&canonical)
            {
                if old == key {
                    return Ok(digest.clone());
                }
            }
        }
        self.active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                canonical.clone(),
                HashProgress {
                    filename: canonical
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    bytes_read: 0,
                    total_bytes: metadata.len(),
                },
            );
        let result = (|| {
            let mut hasher = Sha256::new();
            // Use the reader's standard buffer capacity; no model-size assumptions.
            let mut reader = std::io::BufReader::new(&mut file);
            let mut buffer = vec![0; reader.capacity()];
            loop {
                let count = reader.read(&mut buffer).map_err(io_err)?;
                if count == 0 {
                    break;
                }
                hasher.update(&buffer[..count]);
                if let Some(active) = self
                    .active
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_mut(&canonical)
                {
                    active.bytes_read += count as u64;
                }
            }
            drop(reader);
            // Reject modifications during the read or a path replacement; neither
            // can produce an identity that authorizes different bytes next time.
            if before.is_some()
                && (identity(&file.metadata().map_err(io_err)?) != before
                    || identity(&std::fs::metadata(&canonical).map_err(io_err)?) != before)
            {
                return Err(io_err(std::io::Error::other(
                    "Model file changed while checking it; retry the load.",
                )));
            }
            let digest = hex_lower(&hasher.finalize());
            // HFS+ change times have whole-second precision. Do not cache a
            // file changed in the same clock tick as this read: a same-tick
            // rewrite could otherwise preserve every observable identity field.
            if let Some(key) =
                before.filter(|key| key.changed.0 >= 0 && (key.changed.0 as u64) < started)
            {
                self.completed
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(canonical.clone(), (key, digest.clone()));
            }
            Ok(digest)
        })();
        self.active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&canonical);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Let the file's filesystem change-time tick finish before expecting reuse.
    // Production never sleeps: newly written files simply miss the cache.
    fn finish_change_tick() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_secs(now.as_secs() + 1) - now);
    }

    #[test]
    fn load_hash_matches_uncached_verification_and_reuses_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"original bytes").unwrap();
        finish_change_tick();
        let hashes = LoadHashes::default();
        let digest = hashes.hash(&path).unwrap();
        assert_eq!(digest, super::super::sha256_file_hex(&path).unwrap());
        assert_eq!(hashes.hash(&path).unwrap(), digest);
        assert!(hashes.active.lock().unwrap().is_empty());
        #[cfg(unix)]
        assert_eq!(hashes.completed.lock().unwrap().len(), 1);
    }

    #[test]
    fn same_length_rewrite_with_restored_mtime_cannot_reuse_a_load_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"original bytes").unwrap();
        let original_time = std::fs::metadata(&path).unwrap().modified().unwrap();
        finish_change_tick();
        let hashes = LoadHashes::default();
        let digest = hashes.hash(&path).unwrap();
        std::fs::write(&path, b"modified bytes").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(original_time))
            .unwrap();
        let changed = hashes.hash(&path).unwrap();
        assert_ne!(changed, digest);
        assert_eq!(changed, super::super::sha256_file_hex(&path).unwrap());
        assert!(hashes.active.lock().unwrap().is_empty());
    }

    #[test]
    fn replacement_and_missing_file_do_not_reuse_a_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let hashes = LoadHashes::default();
        std::fs::write(&path, b"first").unwrap();
        let original = hashes.hash(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(hashes.hash(&path).is_err());
        std::fs::write(&path, b"other").unwrap();
        assert_ne!(hashes.hash(&path).unwrap(), original);
        assert!(hashes.active.lock().unwrap().is_empty());
    }
}
