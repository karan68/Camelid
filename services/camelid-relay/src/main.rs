use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use camelid_relay::server::{app, RelayHttpState, RoutePersistence};
use camelid_relay::{RelayError, RelayRouter, UnavailablePush};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StoredRoute {
    route_id: String,
    host_capability: String,
}

struct FileRoutePersistence {
    path: PathBuf,
    routes: Mutex<Vec<StoredRoute>>,
}

impl FileRoutePersistence {
    fn open(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let backup = backup_path(&path);
        if !path.exists() && backup.exists() {
            std::fs::rename(&backup, &path)?;
        }
        let routes = if path.exists() {
            serde_json::from_slice(&std::fs::read(&path)?)?
        } else {
            Vec::new()
        };
        Ok(Self {
            path,
            routes: Mutex::new(routes),
        })
    }

    fn restore_into(&self, router: &RelayRouter) -> Result<(), RelayError> {
        for route in self
            .routes
            .lock()
            .map_err(|_| RelayError::PersistenceUnavailable)?
            .iter()
        {
            router.restore_route(&route.route_id, &route.host_capability)?;
        }
        Ok(())
    }

    fn write(&self, routes: &[StoredRoute]) -> Result<(), RelayError> {
        let parent = self
            .path
            .parent()
            .ok_or(RelayError::PersistenceUnavailable)?;
        std::fs::create_dir_all(parent).map_err(|_| RelayError::PersistenceUnavailable)?;
        let temporary = parent.join(format!(".relay-routes-{}.tmp", std::process::id()));
        let backup = backup_path(&self.path);
        let encoded = serde_json::to_vec(routes).map_err(|_| RelayError::PersistenceUnavailable)?;
        std::fs::write(&temporary, encoded).map_err(|_| RelayError::PersistenceUnavailable)?;
        secure_file(&temporary)?;
        if self.path.exists() {
            let _ = std::fs::remove_file(&backup);
            std::fs::rename(&self.path, &backup).map_err(|_| RelayError::PersistenceUnavailable)?;
        }
        if std::fs::rename(&temporary, &self.path).is_err() {
            if backup.exists() {
                let _ = std::fs::rename(&backup, &self.path);
            }
            let _ = std::fs::remove_file(&temporary);
            return Err(RelayError::PersistenceUnavailable);
        }
        let _ = std::fs::remove_file(backup);
        Ok(())
    }
}

impl RoutePersistence for FileRoutePersistence {
    fn insert(&self, route_id: &str, host_capability: &str) -> Result<(), RelayError> {
        let mut routes = self
            .routes
            .lock()
            .map_err(|_| RelayError::PersistenceUnavailable)?;
        if routes.iter().any(|route| route.route_id == route_id) {
            return Err(RelayError::PersistenceUnavailable);
        }
        let next = StoredRoute {
            route_id: route_id.into(),
            host_capability: host_capability.into(),
        };
        routes.push(next);
        if let Err(error) = self.write(&routes) {
            routes.pop();
            return Err(error);
        }
        Ok(())
    }
}

fn backup_path(path: &Path) -> PathBuf {
    path.with_extension("bak")
}

#[cfg(unix)]
fn secure_file(path: &Path) -> Result<(), RelayError> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|_| RelayError::PersistenceUnavailable)
}

#[cfg(not(unix))]
fn secure_file(_: &Path) -> Result<(), RelayError> {
    Err(RelayError::PersistenceUnavailable)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let address: SocketAddr = std::env::var("CAMELID_RELAY_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8787".into())
        .parse()?;
    let enrollment_token = std::env::var("CAMELID_RELAY_ENROLLMENT_TOKEN")
        .map_err(|_| "CAMELID_RELAY_ENROLLMENT_TOKEN is required")?;
    let keepalive_ms: u64 = std::env::var("CAMELID_RELAY_KEEPALIVE_MS")
        .map_err(|_| "CAMELID_RELAY_KEEPALIVE_MS is required")?
        .parse()?;
    if keepalive_ms == 0 {
        return Err("CAMELID_RELAY_KEEPALIVE_MS must be at least 1".into());
    }
    let router = RelayRouter::new(Arc::new(UnavailablePush));
    let mut state = RelayHttpState::new(router.clone(), enrollment_token)?
        .with_keepalive_interval(Duration::from_millis(keepalive_ms));
    if let Some(path) = std::env::var_os("CAMELID_RELAY_STATE").map(PathBuf::from) {
        let persistence = Arc::new(FileRoutePersistence::open(path)?);
        persistence.restore_into(&router)?;
        state = state.with_route_persistence(persistence);
    }
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("camelid relay listening on {}", listener.local_addr()?);
    axum::serve(listener, app(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn file_route_store_restores_exact_credentials_after_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("routes.json");
        let source_router = RelayRouter::new(Arc::new(UnavailablePush));
        let enrolled = source_router.enroll_route().unwrap();
        let route_id = enrolled.route_id.expose_for_enrollment();
        let host_capability = enrolled.host_capability.expose_for_enrollment();
        let first = FileRoutePersistence::open(path.clone()).unwrap();
        first.insert(&route_id, &host_capability).unwrap();
        drop(first);

        let restored = FileRoutePersistence::open(path).unwrap();
        let router = RelayRouter::new(Arc::new(UnavailablePush));
        restored.restore_into(&router).unwrap();
        let route = camelid_relay::RouteId::parse(&route_id).unwrap();
        let capability = camelid_relay::HostCapability::parse(&host_capability).unwrap();
        assert!(router.connect_host(route, capability).is_ok());
    }

    #[cfg(not(unix))]
    #[test]
    fn route_file_persistence_fails_closed_without_mode_0600() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("routes.json");
        let store = FileRoutePersistence::open(path).unwrap();
        assert!(matches!(
            store.insert("AAAAAAAAAAAAAAAAAAAAAA", "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"),
            Err(RelayError::PersistenceUnavailable)
        ));
    }
}
