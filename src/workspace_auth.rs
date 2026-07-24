use std::fs::OpenOptions;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

const FILE_VERSION: &str = "camelid.workspace-cli/v1";
const TOKEN_FILE_ENV: &str = "CAMELID_WORKSPACE_TOKEN_FILE";

pub(crate) struct WorkspaceCliCredential {
    path: PathBuf,
    token: String,
}

impl WorkspaceCliCredential {
    pub(crate) fn issue(addr: SocketAddr) -> io::Result<Self> {
        let path = token_path(addr)?;
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid token path"))?;
        std::fs::create_dir_all(parent)?;

        let mut secret = [0_u8; 32];
        getrandom::fill(&mut secret).map_err(|error| io::Error::other(error.to_string()))?;
        let token = secret
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let temporary = parent.join(format!(
            ".{}.{}.tmp",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("workspace-token"),
            uuid::Uuid::new_v4().simple()
        ));
        if let Err(error) = write_private(&temporary, &format!("{FILE_VERSION}\n{token}\n")) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error);
        }

        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        if let Err(error) = std::fs::rename(&temporary, &path) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error);
        }

        Ok(Self { path, token })
    }

    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for WorkspaceCliCredential {
    fn drop(&mut self) {
        if read_token(&self.path).is_ok_and(|token| token == self.token) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub(crate) fn load_token(addr: SocketAddr) -> io::Result<String> {
    read_token(&token_path(addr)?)
}

pub(crate) fn token_matches(expected: &str, provided: &str) -> bool {
    if expected.len() != provided.len() {
        return false;
    }
    expected
        .as_bytes()
        .iter()
        .zip(provided.as_bytes())
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn read_token(path: &Path) -> io::Result<String> {
    let contents = std::fs::read_to_string(path)?;
    let mut lines = contents.lines();
    if lines.next() != Some(FILE_VERSION) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported Workspace CLI credential format",
        ));
    }
    let token = lines.next().unwrap_or_default();
    if token.len() != 64 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Workspace CLI credential",
        ));
    }
    Ok(token.to_string())
}

fn token_path(addr: SocketAddr) -> io::Result<PathBuf> {
    if let Some(path) = std::env::var_os(TOKEN_FILE_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }

    let mut base = runtime_base_dir()?;
    base.push("camelid");
    base.push("runtime");
    let host = addr
        .ip()
        .to_string()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    base.push(format!("workspace-{host}-{}.token", addr.port()));
    Ok(base)
}

#[cfg(windows)]
fn runtime_base_dir() -> io::Result<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "LOCALAPPDATA is not set"))
}

#[cfg(not(windows))]
fn runtime_base_dir() -> io::Result<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_RUNTIME_DIR").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(|home| PathBuf::from(home).join(".cache"))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "neither XDG_RUNTIME_DIR nor HOME is set",
            )
        })
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
    fn credential_round_trips_and_is_removed_on_drop() {
        let _environment = crate::test_support::env_lock();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("workspace.token");
        std::env::set_var(TOKEN_FILE_ENV, &path);

        let credential = WorkspaceCliCredential::issue("127.0.0.1:8181".parse().unwrap()).unwrap();
        assert_eq!(credential.path(), path);
        assert_eq!(
            load_token("127.0.0.1:8181".parse().unwrap()).unwrap(),
            credential.token()
        );
        assert!(token_matches(credential.token(), credential.token()));
        assert!(!token_matches(credential.token(), &"0".repeat(64)));
        drop(credential);
        assert!(!path.exists());

        std::env::remove_var(TOKEN_FILE_ENV);
    }

    #[test]
    fn stale_credential_does_not_delete_a_replacement() {
        let _environment = crate::test_support::env_lock();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("workspace.token");
        std::env::set_var(TOKEN_FILE_ENV, &path);

        let old = WorkspaceCliCredential::issue("127.0.0.1:8181".parse().unwrap()).unwrap();
        let replacement = WorkspaceCliCredential::issue("127.0.0.1:8181".parse().unwrap()).unwrap();
        drop(old);
        assert_eq!(read_token(&path).unwrap(), replacement.token());
        drop(replacement);
        assert!(!path.exists());

        std::env::remove_var(TOKEN_FILE_ENV);
    }

    #[cfg(unix)]
    #[test]
    fn credential_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let _environment = crate::test_support::env_lock();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("workspace.token");
        std::env::set_var(TOKEN_FILE_ENV, &path);
        let credential = WorkspaceCliCredential::issue("127.0.0.1:8181".parse().unwrap()).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        drop(credential);
        std::env::remove_var(TOKEN_FILE_ENV);
    }
}
