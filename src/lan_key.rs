//! Local operator credential for authenticated LAN Chat.

use std::fmt;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const KEY_FILE_NAME: &str = "lan-chat.key";
const KEY_HEX_BYTES: usize = 64;
const MAX_KEY_BYTES: usize = 4 * 1024;

pub struct LanChatKey {
    path: PathBuf,
    secret: String,
    created: bool,
}

impl fmt::Debug for LanChatKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LanChatKey")
            .field("path", &self.path)
            .field("secret", &"[REDACTED]")
            .field("created", &self.created)
            .finish()
    }
}

impl LanChatKey {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn secret(&self) -> &str {
        &self.secret
    }

    pub fn created(&self) -> bool {
        self.created
    }
}

pub fn provision(rotate: bool) -> io::Result<LanChatKey> {
    provision_at(default_path()?, rotate)
}

pub fn default_path() -> io::Result<PathBuf> {
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "LOCALAPPDATA is not set"))?;

    #[cfg(not(windows))]
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|value| !value.is_empty())
                .map(|home| PathBuf::from(home).join(".config"))
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "neither XDG_CONFIG_HOME nor HOME is set",
            )
        })?;

    Ok(base.join("Camelid").join(KEY_FILE_NAME))
}

fn provision_at(path: PathBuf, rotate: bool) -> io::Result<LanChatKey> {
    if path.exists() && !rotate {
        return Ok(LanChatKey {
            secret: read_key(&path)?,
            path,
            created: false,
        });
    }

    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid key path"))?;
    std::fs::create_dir_all(parent)?;
    let mut random = [0_u8; 32];
    getrandom::fill(&mut random).map_err(|error| io::Error::other(error.to_string()))?;
    let secret = random
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    debug_assert_eq!(secret.len(), KEY_HEX_BYTES);
    let temporary = parent.join(format!(
        ".{KEY_FILE_NAME}.{}.tmp",
        uuid::Uuid::new_v4().simple()
    ));
    if let Err(error) = write_private(&temporary, &format!("{secret}\n")) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    if !rotate {
        match std::fs::hard_link(&temporary, &path) {
            Ok(()) => {
                std::fs::remove_file(&temporary)?;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let _ = std::fs::remove_file(&temporary);
                return Ok(LanChatKey {
                    secret: read_key(&path)?,
                    path,
                    created: false,
                });
            }
            Err(error) => {
                let _ = std::fs::remove_file(&temporary);
                return Err(error);
            }
        }
    } else if path.exists() {
        reject_symlink_or_non_file(&path)?;
        let backup = parent.join(format!(
            ".{KEY_FILE_NAME}.{}.backup",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::rename(&path, &backup)?;
        if let Err(error) = std::fs::rename(&temporary, &path) {
            let _ = std::fs::rename(&backup, &path);
            let _ = std::fs::remove_file(&temporary);
            return Err(error);
        }
        let _ = std::fs::remove_file(backup);
    } else if let Err(error) = std::fs::rename(&temporary, &path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(LanChatKey {
        path,
        secret,
        created: true,
    })
}

fn read_key(path: &Path) -> io::Result<String> {
    reject_symlink_or_non_file(path)?;
    let key = std::fs::read_to_string(path)?.trim().to_string();
    if key.is_empty() || key.len() > MAX_KEY_BYTES || key.chars().any(char::is_control) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "LAN Chat key file is invalid; rotate it explicitly",
        ));
    }
    Ok(key)
}

fn reject_symlink_or_non_file(path: &Path) -> io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "LAN Chat key path must be a direct regular file",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn write_private(path: &Path, contents: &str) -> io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(contents.as_bytes())?;
    file.sync_all()
}

#[cfg(windows)]
fn write_private(path: &Path, contents: &str) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(contents.as_bytes())?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provision_reuses_and_rotates_only_when_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(KEY_FILE_NAME);
        let first = provision_at(path.clone(), false).unwrap();
        assert!(first.created());
        assert_eq!(first.secret().len(), KEY_HEX_BYTES);
        assert!(!format!("{first:?}").contains(first.secret()));

        let reused = provision_at(path.clone(), false).unwrap();
        assert!(!reused.created());
        assert_eq!(reused.secret(), first.secret());

        let rotated = provision_at(path.clone(), true).unwrap();
        assert!(rotated.created());
        assert_ne!(rotated.secret(), first.secret());
        assert_eq!(read_key(&path).unwrap(), rotated.secret());
    }

    #[test]
    fn malformed_existing_key_fails_instead_of_silently_rotating() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(KEY_FILE_NAME);
        std::fs::write(&path, "\n").unwrap();
        assert_eq!(
            provision_at(path, false).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn concurrent_first_use_converges_on_one_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(KEY_FILE_NAME);
        let threads = (0..8)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || provision_at(path, false).unwrap().secret)
            })
            .collect::<Vec<_>>();
        let keys = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert!(keys.iter().all(|key| key == &keys[0]));
        assert_eq!(read_key(&path).unwrap(), keys[0]);
    }

    #[test]
    fn an_existing_symlink_is_never_read_or_rotated() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        let path = dir.path().join(KEY_FILE_NAME);
        std::fs::write(&target, "a-valid-looking-key").unwrap();

        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&target, &path).is_ok();
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_file(&target, &path).is_ok();

        if linked {
            assert_eq!(
                provision_at(path.clone(), false).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
            assert_eq!(
                provision_at(path, true).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
            assert_eq!(
                std::fs::read_to_string(target).unwrap(),
                "a-valid-looking-key"
            );
        }
    }
}
