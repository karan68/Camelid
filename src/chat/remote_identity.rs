use std::path::{Path, PathBuf};

use camelid_remote_crypto::StaticKeypair;
use camelid_remote_store::{RemoteStore, StoreError};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum HostIdentityError {
    #[error("host secret storage is unavailable")]
    SecretStoreUnavailable,
    #[error("host identity is invalid")]
    InvalidIdentity,
    #[error("host identity persistence is unavailable")]
    PersistenceUnavailable,
}

pub trait HostSecretStore {
    fn store(&self, reference: &str, secret: &[u8; 32]) -> Result<(), HostIdentityError>;
    fn load(&self, reference: &str) -> Result<[u8; 32], HostIdentityError>;
    fn delete(&self, reference: &str) -> Result<(), HostIdentityError>;
}

pub struct HostIdentity {
    pub host_id: Uuid,
    pub public_key: [u8; 32],
    private_key: [u8; 32],
}

impl HostIdentity {
    pub fn private_key(&self) -> &[u8; 32] {
        &self.private_key
    }
}

impl Drop for HostIdentity {
    fn drop(&mut self) {
        self.private_key.fill(0);
    }
}

pub fn load_or_create(
    store: &mut RemoteStore,
    secrets: &dyn HostSecretStore,
    created_at_unix_ms: u64,
) -> Result<HostIdentity, HostIdentityError> {
    if let Some(stored) = store.optional_host_identity().map_err(map_store_error)? {
        let private_key = secrets.load(&stored.secret_reference)?;
        return Ok(HostIdentity {
            host_id: stored.host_id,
            public_key: stored.noise_public,
            private_key,
        });
    }

    let keypair = StaticKeypair::generate().map_err(|_| HostIdentityError::InvalidIdentity)?;
    let host_id = Uuid::new_v4();
    let secret_reference = format!("dpapi-file:v1:{}", Uuid::new_v4());
    secrets.store(&secret_reference, keypair.private())?;
    if store
        .initialize_host_identity(
            host_id,
            keypair.public(),
            &secret_reference,
            created_at_unix_ms,
        )
        .is_err()
    {
        let _ = secrets.delete(&secret_reference);
        return Err(HostIdentityError::PersistenceUnavailable);
    }
    Ok(HostIdentity {
        host_id,
        public_key: *keypair.public(),
        private_key: *keypair.private(),
    })
}

fn map_store_error(_: StoreError) -> HostIdentityError {
    HostIdentityError::PersistenceUnavailable
}

pub struct ProtectedFileSecretStore {
    root: PathBuf,
}

impl ProtectedFileSecretStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn path_for(&self, reference: &str) -> Result<PathBuf, HostIdentityError> {
        let id = reference
            .strip_prefix("dpapi-file:v1:")
            .and_then(|value| Uuid::parse_str(value).ok())
            .ok_or(HostIdentityError::InvalidIdentity)?;
        Ok(self.root.join(format!("{id}.key")))
    }

    pub fn store_bytes(&self, reference: &str, secret: &[u8]) -> Result<(), HostIdentityError> {
        if secret.is_empty() || secret.len() > 4096 {
            return Err(HostIdentityError::InvalidIdentity);
        }
        let path = self.path_for(reference)?;
        let mut protected = protect(secret)?;
        let result = write_secret_file(&path, &protected);
        protected.fill(0);
        result
    }

    pub fn load_bytes(&self, reference: &str) -> Result<Vec<u8>, HostIdentityError> {
        let path = self.path_for(reference)?;
        let protected =
            std::fs::read(path).map_err(|_| HostIdentityError::SecretStoreUnavailable)?;
        unprotect(&protected)
    }

    pub fn delete_bytes(&self, reference: &str) -> Result<(), HostIdentityError> {
        let path = self.path_for(reference)?;
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(_) => Err(HostIdentityError::SecretStoreUnavailable),
        }
    }
}

impl HostSecretStore for ProtectedFileSecretStore {
    fn store(&self, reference: &str, secret: &[u8; 32]) -> Result<(), HostIdentityError> {
        self.store_bytes(reference, secret)
    }

    fn load(&self, reference: &str) -> Result<[u8; 32], HostIdentityError> {
        self.load_bytes(reference)?
            .try_into()
            .map_err(|_| HostIdentityError::InvalidIdentity)
    }

    fn delete(&self, reference: &str) -> Result<(), HostIdentityError> {
        self.delete_bytes(reference)
    }
}

fn write_secret_file(path: &Path, bytes: &[u8]) -> Result<(), HostIdentityError> {
    use std::io::Write;

    let parent = path
        .parent()
        .ok_or(HostIdentityError::SecretStoreUnavailable)?;
    std::fs::create_dir_all(parent).map_err(|_| HostIdentityError::SecretStoreUnavailable)?;
    let temporary = parent.join(format!(".{}.tmp", Uuid::new_v4()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|_| HostIdentityError::SecretStoreUnavailable)?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| HostIdentityError::SecretStoreUnavailable)?;
        if path.exists() {
            return Err(HostIdentityError::InvalidIdentity);
        }
        std::fs::rename(&temporary, path).map_err(|_| HostIdentityError::SecretStoreUnavailable)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
    result
}

#[cfg(windows)]
fn protect(secret: &[u8]) -> Result<Vec<u8>, HostIdentityError> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    let input = CRYPT_INTEGER_BLOB {
        cbData: secret.len() as u32,
        pbData: secret.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let succeeded = unsafe {
        CryptProtectData(
            &input,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if succeeded == 0 || output.pbData.is_null() {
        return Err(HostIdentityError::SecretStoreUnavailable);
    }
    let protected = unsafe {
        let bytes = std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec();
        LocalFree(output.pbData.cast());
        bytes
    };
    Ok(protected)
}

#[cfg(windows)]
fn unprotect(protected: &[u8]) -> Result<Vec<u8>, HostIdentityError> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    let input = CRYPT_INTEGER_BLOB {
        cbData: protected
            .len()
            .try_into()
            .map_err(|_| HostIdentityError::InvalidIdentity)?,
        pbData: protected.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let succeeded = unsafe {
        CryptUnprotectData(
            &input,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if succeeded == 0 || output.pbData.is_null() || output.cbData == 0 || output.cbData > 4096 {
        if !output.pbData.is_null() {
            unsafe { LocalFree(output.pbData.cast()) };
        }
        return Err(HostIdentityError::InvalidIdentity);
    }
    let secret_len = output.cbData as usize;
    let mut secret = vec![0_u8; secret_len];
    unsafe {
        secret.copy_from_slice(std::slice::from_raw_parts(output.pbData, secret_len));
        std::ptr::write_bytes(output.pbData, 0, secret_len);
        LocalFree(output.pbData.cast());
    }
    Ok(secret)
}

#[cfg(not(windows))]
fn protect(_: &[u8]) -> Result<Vec<u8>, HostIdentityError> {
    Err(HostIdentityError::SecretStoreUnavailable)
}

#[cfg(not(windows))]
fn unprotect(_: &[u8]) -> Result<Vec<u8>, HostIdentityError> {
    Err(HostIdentityError::SecretStoreUnavailable)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct MemorySecretStore {
        values: Mutex<HashMap<String, [u8; 32]>>,
    }

    impl HostSecretStore for MemorySecretStore {
        fn store(&self, reference: &str, secret: &[u8; 32]) -> Result<(), HostIdentityError> {
            let mut values = self.values.lock().unwrap();
            if values.insert(reference.to_string(), *secret).is_some() {
                return Err(HostIdentityError::InvalidIdentity);
            }
            Ok(())
        }

        fn load(&self, reference: &str) -> Result<[u8; 32], HostIdentityError> {
            self.values
                .lock()
                .unwrap()
                .get(reference)
                .copied()
                .ok_or(HostIdentityError::SecretStoreUnavailable)
        }

        fn delete(&self, reference: &str) -> Result<(), HostIdentityError> {
            self.values.lock().unwrap().remove(reference);
            Ok(())
        }
    }

    #[test]
    fn identity_is_created_once_and_private_key_stays_out_of_sqlite() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("remote.sqlite3");
        let mut store = RemoteStore::open(&database).unwrap();
        let secrets = MemorySecretStore::default();

        let first = load_or_create(&mut store, &secrets, 1).unwrap();
        let first_private = *first.private_key();
        let first_public = first.public_key;
        let first_id = first.host_id;
        drop(first);
        let loaded = load_or_create(&mut store, &secrets, 2).unwrap();

        assert_eq!(loaded.host_id, first_id);
        assert_eq!(loaded.public_key, first_public);
        assert_eq!(loaded.private_key(), &first_private);
        let database_bytes = std::fs::read(database).unwrap();
        assert!(!database_bytes
            .windows(first_private.len())
            .any(|window| window == first_private));
    }

    #[cfg(windows)]
    #[test]
    fn protected_file_store_encrypts_for_the_current_windows_user() {
        let directory = tempfile::tempdir().unwrap();
        let secrets = ProtectedFileSecretStore::new(directory.path().to_path_buf());
        let reference = format!("dpapi-file:v1:{}", Uuid::new_v4());
        let secret = [0xA5_u8; 32];

        secrets.store(&reference, &secret).unwrap();
        assert_eq!(secrets.load(&reference).unwrap(), secret);
        let encrypted = std::fs::read(secrets.path_for(&reference).unwrap()).unwrap();
        assert!(!encrypted
            .windows(secret.len())
            .any(|window| window == secret));
        secrets.delete(&reference).unwrap();
        assert!(!secrets.path_for(&reference).unwrap().exists());
    }
}
