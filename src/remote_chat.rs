//! Private cross-network access to an authenticated LAN Chat listener.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context};
use serde::{Deserialize, Serialize};

use crate::chat::client::Client;

const DEFAULT_BACKEND: &str = "127.0.0.1:8181";
const HTTPS_PORT: u16 = 443;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
const SERVE_CREATION_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_COMMAND_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const CONFIG_VERIFY_ATTEMPTS: usize = 4;
const CONFIG_VERIFY_DELAY: Duration = Duration::from_millis(100);
const MIN_TAILSCALE_VERSION: (u32, u32) = (1, 52);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportState {
    Inactive,
    Active,
    Conflict,
    PublicFunnel,
}

impl fmt::Display for TransportState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Inactive => "inactive",
            Self::Active => "active (tailnet only)",
            Self::Conflict => "conflict",
            Self::PublicFunnel => "public Funnel",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RemoteChatStatus {
    pub backend: SocketAddr,
    pub url: String,
    pub backend_ready: bool,
    pub transport: TransportState,
}

#[derive(Clone, Debug)]
pub struct RemoteChatOptions {
    pub backend: SocketAddr,
    pub tailscale_bin: Option<PathBuf>,
}

impl Default for RemoteChatOptions {
    fn default() -> Self {
        Self {
            backend: DEFAULT_BACKEND.parse().expect("valid default backend"),
            tailscale_bin: None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TailscaleStatus {
    #[serde(rename = "BackendState")]
    backend_state: String,
    #[serde(rename = "Self")]
    self_node: Option<TailscaleSelf>,
}

#[derive(Debug, Deserialize)]
struct TailscaleSelf {
    #[serde(rename = "DNSName")]
    dns_name: String,
    #[serde(rename = "Online", default)]
    online: bool,
}

#[derive(Debug, Default, Deserialize)]
struct ServeConfig {
    #[serde(rename = "TCP", default)]
    tcp: HashMap<String, TcpPortHandler>,
    #[serde(rename = "Web", default)]
    web: HashMap<String, WebServerConfig>,
    #[serde(rename = "AllowFunnel", default)]
    allow_funnel: HashMap<String, bool>,
    #[serde(rename = "Foreground", default)]
    foreground: HashMap<String, ServeConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct TcpPortHandler {
    #[serde(rename = "HTTPS", default)]
    https: bool,
    #[serde(rename = "HTTP", default)]
    http: bool,
    #[serde(rename = "TCPForward", default)]
    tcp_forward: String,
    #[serde(rename = "TerminateTLS", default)]
    terminate_tls: String,
}

#[derive(Debug, Default, Deserialize)]
struct WebServerConfig {
    #[serde(rename = "Handlers", default)]
    handlers: HashMap<String, HttpHandler>,
}

#[derive(Debug, Default, Deserialize)]
struct HttpHandler {
    #[serde(rename = "Proxy", default)]
    proxy: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TailnetIdentity {
    dns_name: String,
}

#[derive(Debug)]
struct ConfigInspection {
    state: TransportState,
    owns_exact_mapping: bool,
}

#[derive(Debug)]
struct CapturedOutput {
    success: bool,
    code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

trait TailscaleRunner {
    fn run(&self, args: &[&str]) -> anyhow::Result<CapturedOutput>;
}

struct TailscaleCli {
    executable: PathBuf,
}

impl TailscaleCli {
    fn discover(explicit: Option<&Path>) -> anyhow::Result<Self> {
        let cli = Self {
            executable: resolve_tailscale_bin(explicit)?,
        };
        cli.require_supported_version()?;
        Ok(cli)
    }

    fn require_supported_version(&self) -> anyhow::Result<()> {
        let raw = run_checked(self, &["version"], "read Tailscale CLI version")?;
        let version = parse_tailscale_version(&raw)?;
        ensure!(
            version >= MIN_TAILSCALE_VERSION,
            "Tailscale {}.{} or newer is required; found {}.{}",
            MIN_TAILSCALE_VERSION.0,
            MIN_TAILSCALE_VERSION.1,
            version.0,
            version.1
        );
        Ok(())
    }
}

impl TailscaleRunner for TailscaleCli {
    fn run(&self, args: &[&str]) -> anyhow::Result<CapturedOutput> {
        run_process(&self.executable, args, command_timeout(args))
    }
}

fn command_timeout(args: &[&str]) -> Duration {
    if matches!(args, ["serve", "--bg", "--yes", "--https=443", _]) {
        SERVE_CREATION_TIMEOUT
    } else {
        COMMAND_TIMEOUT
    }
}

impl TailnetIdentity {
    pub fn url(&self) -> String {
        format!("https://{}/", self.dns_name)
    }

    fn host_port(&self) -> String {
        format!("{}:{HTTPS_PORT}", self.dns_name)
    }
}

fn validate_backend_addr(backend: SocketAddr) -> anyhow::Result<()> {
    ensure!(
        backend.ip() == Ipv4Addr::LOCALHOST,
        "remote Chat only proxies an IPv4 loopback listener at 127.0.0.1; got {backend}"
    );
    Ok(())
}

fn require_lan_chat_backend(backend: SocketAddr) -> anyhow::Result<()> {
    validate_backend_addr(backend)?;
    let health = Client::new(backend).health().with_context(|| {
        format!(
            "no healthy Camelid listener answered at http://{backend}; start `camelid serve --lan-chat-only --api-key-file <PATH> --addr {backend}` first"
        )
    })?;
    ensure!(
        health.api_surface.as_deref() == Some("lan_chat_only"),
        "refusing to publish the listener at {backend}: /v1/health did not report api_surface=lan_chat_only"
    );
    Ok(())
}

fn lan_chat_backend_ready(backend: SocketAddr) -> bool {
    validate_backend_addr(backend).is_ok()
        && Client::new(backend)
            .health()
            .is_some_and(|health| health.api_surface.as_deref() == Some("lan_chat_only"))
}

fn parse_tailnet_identity(raw: &[u8]) -> anyhow::Result<TailnetIdentity> {
    let status: TailscaleStatus =
        serde_json::from_slice(raw).context("`tailscale status --json` returned invalid JSON")?;
    ensure!(
        status.backend_state == "Running",
        "Tailscale is not connected (state: {})",
        status.backend_state
    );
    let self_node = status
        .self_node
        .context("Tailscale status did not identify this device")?;
    ensure!(self_node.online, "this Tailscale device is offline");
    let dns_name = self_node.dns_name.trim_end_matches('.').to_string();
    ensure!(
        valid_dns_name(&dns_name),
        "Tailscale returned an invalid device DNS name"
    );
    Ok(TailnetIdentity { dns_name })
}

fn valid_dns_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 253
        && name.is_ascii()
        && name.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        })
}

fn parse_tailscale_version(raw: &[u8]) -> anyhow::Result<(u32, u32)> {
    let decoded = String::from_utf8_lossy(raw);
    let line = decoded
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .context("`tailscale version` returned no version")?;
    let mut components = line.trim_start_matches('v').split('.');
    let major = components
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .context("`tailscale version` returned an invalid major version")?;
    let minor = components
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .context("`tailscale version` returned an invalid minor version")?;
    Ok((major, minor))
}

#[cfg(test)]
fn inspect_serve_config(
    raw: &[u8],
    identity: &TailnetIdentity,
    backend: SocketAddr,
) -> anyhow::Result<TransportState> {
    validate_backend_addr(backend)?;
    let config: ServeConfig = serde_json::from_slice(raw)
        .context("`tailscale serve status --json` returned invalid JSON")?;
    Ok(inspect_config(&config, identity, &proxy_target(backend)).state)
}

fn proxy_target(backend: SocketAddr) -> String {
    format!("http://{backend}")
}

fn inspect_config(
    config: &ServeConfig,
    identity: &TailnetIdentity,
    target: &str,
) -> ConfigInspection {
    let host_port = identity.host_port();
    let owns_exact_mapping = config
        .web
        .iter()
        .filter(|(entry, _)| port_is(entry, HTTPS_PORT))
        .count()
        == 1
        && config.web.get(&host_port).is_some_and(|web| {
            web.handlers.len() == 1
                && web
                    .handlers
                    .get("/")
                    .is_some_and(|handler| handler.proxy == target)
        })
        && config
            .tcp
            .get(&HTTPS_PORT.to_string())
            .is_some_and(|handler| {
                handler.https
                    && !handler.http
                    && handler.tcp_forward.is_empty()
                    && handler.terminate_tls.is_empty()
            });

    if config
        .allow_funnel
        .iter()
        .any(|(host_port, enabled)| *enabled && port_is(host_port, HTTPS_PORT))
    {
        return ConfigInspection {
            state: TransportState::PublicFunnel,
            owns_exact_mapping,
        };
    }

    if config
        .foreground
        .values()
        .any(|foreground| config_uses_port(foreground, HTTPS_PORT))
    {
        return ConfigInspection {
            state: TransportState::Conflict,
            owns_exact_mapping,
        };
    }

    if owns_exact_mapping {
        return ConfigInspection {
            state: TransportState::Active,
            owns_exact_mapping,
        };
    }
    if config_uses_port(config, HTTPS_PORT) {
        return ConfigInspection {
            state: TransportState::Conflict,
            owns_exact_mapping,
        };
    }
    ConfigInspection {
        state: TransportState::Inactive,
        owns_exact_mapping,
    }
}

fn config_uses_port(config: &ServeConfig, port: u16) -> bool {
    config.tcp.contains_key(&port.to_string())
        || config.web.keys().any(|host_port| port_is(host_port, port))
}

fn port_is(host_port: &str, expected: u16) -> bool {
    host_port
        .rsplit_once(':')
        .and_then(|(_, port)| port.parse::<u16>().ok())
        == Some(expected)
}

fn require_startable(state: TransportState) -> anyhow::Result<bool> {
    match state {
        TransportState::Inactive => Ok(true),
        TransportState::Active => Ok(false),
        TransportState::Conflict => {
            bail!("Tailscale HTTPS port {HTTPS_PORT} already serves another target; inspect `tailscale serve status` and move or remove that mapping before retrying; Camelid did not change it")
        }
        TransportState::PublicFunnel => {
            bail!("Tailscale Funnel is enabled on HTTPS port {HTTPS_PORT}; disable it with `tailscale funnel --https=443 off` before retrying; Camelid refuses to publish remote Chat to the public internet")
        }
    }
}

pub fn start(options: RemoteChatOptions) -> anyhow::Result<RemoteChatStatus> {
    require_lan_chat_backend(options.backend)?;
    let runner = TailscaleCli::discover(options.tailscale_bin.as_deref())?;
    start_transport(options.backend, &runner)
}

pub fn status(options: RemoteChatOptions) -> anyhow::Result<RemoteChatStatus> {
    validate_backend_addr(options.backend)?;
    let runner = TailscaleCli::discover(options.tailscale_bin.as_deref())?;
    status_transport(options.backend, &runner)
}

pub fn stop(options: RemoteChatOptions) -> anyhow::Result<RemoteChatStatus> {
    validate_backend_addr(options.backend)?;
    let runner = TailscaleCli::discover(options.tailscale_bin.as_deref())?;
    stop_transport(options.backend, &runner)
}

fn start_transport(
    backend: SocketAddr,
    runner: &impl TailscaleRunner,
) -> anyhow::Result<RemoteChatStatus> {
    let identity = read_identity(runner)?;
    let before = read_config(runner, &identity, backend)?;
    if require_startable(before.state)? {
        let target = proxy_target(backend);
        run_checked(
            runner,
            &["serve", "--bg", "--yes", "--https=443", &target],
            "enable private Tailscale Serve",
        )?;

        let verified = read_config_until(runner, &identity, backend, |inspection| {
            inspection.state == TransportState::Active
        });
        match verified {
            Ok(inspection) if inspection.state == TransportState::Active => {}
            Ok(inspection) => {
                let rollback = disable_mapping(runner);
                let suffix = rollback_suffix(&rollback);
                bail!(
                    "Tailscale accepted the Serve command but the Camelid mapping was not active afterward (state: {:?}){suffix}",
                    inspection.state
                );
            }
            Err(error) => {
                let rollback = disable_mapping(runner);
                let suffix = rollback_suffix(&rollback);
                return Err(error.context(format!(
                    "could not verify the Tailscale Serve mapping after creation{suffix}"
                )));
            }
        }
    }

    Ok(RemoteChatStatus {
        backend,
        url: identity.url(),
        backend_ready: true,
        transport: TransportState::Active,
    })
}

fn status_transport(
    backend: SocketAddr,
    runner: &impl TailscaleRunner,
) -> anyhow::Result<RemoteChatStatus> {
    let identity = read_identity(runner)?;
    let inspection = read_config(runner, &identity, backend)?;
    Ok(RemoteChatStatus {
        backend,
        url: identity.url(),
        backend_ready: lan_chat_backend_ready(backend),
        transport: inspection.state,
    })
}

fn stop_transport(
    backend: SocketAddr,
    runner: &impl TailscaleRunner,
) -> anyhow::Result<RemoteChatStatus> {
    let identity = read_identity(runner)?;
    let before = read_config(runner, &identity, backend)?;
    if before.state == TransportState::Active {
        debug_assert!(before.owns_exact_mapping);
        disable_mapping(runner)?;
        let after = read_config_until(runner, &identity, backend, |inspection| {
            inspection.state != TransportState::Active
        })?;
        ensure!(
            after.state != TransportState::Active,
            "Tailscale accepted the disable command but the Camelid mapping is still active"
        );
        return Ok(RemoteChatStatus {
            backend,
            url: identity.url(),
            backend_ready: lan_chat_backend_ready(backend),
            transport: after.state,
        });
    }

    match before.state {
        TransportState::Inactive => {}
        TransportState::Conflict => {
            bail!("Tailscale HTTPS port {HTTPS_PORT} belongs to another target; inspect `tailscale serve status` before changing it; Camelid did not change it")
        }
        TransportState::PublicFunnel => {
            bail!("Tailscale Funnel is public on HTTPS port {HTTPS_PORT}, but it does not point to this Camelid listener; inspect `tailscale funnel status`; Camelid did not change it")
        }
        TransportState::Active => unreachable!("active mapping handled above"),
    }
    Ok(RemoteChatStatus {
        backend,
        url: identity.url(),
        backend_ready: lan_chat_backend_ready(backend),
        transport: TransportState::Inactive,
    })
}

fn read_identity(runner: &impl TailscaleRunner) -> anyhow::Result<TailnetIdentity> {
    let raw = run_checked(runner, &["status", "--json"], "read Tailscale status")?;
    parse_tailnet_identity(&raw)
}

fn read_config(
    runner: &impl TailscaleRunner,
    identity: &TailnetIdentity,
    backend: SocketAddr,
) -> anyhow::Result<ConfigInspection> {
    let raw = run_checked(
        runner,
        &["serve", "status", "--json"],
        "read Tailscale Serve status",
    )?;
    let config: ServeConfig = serde_json::from_slice(&raw)
        .context("`tailscale serve status --json` returned invalid JSON")?;
    Ok(inspect_config(&config, identity, &proxy_target(backend)))
}

fn read_config_until(
    runner: &impl TailscaleRunner,
    identity: &TailnetIdentity,
    backend: SocketAddr,
    ready: impl Fn(&ConfigInspection) -> bool,
) -> anyhow::Result<ConfigInspection> {
    let mut last_inspection = None;
    let mut last_error = None;
    for attempt in 0..CONFIG_VERIFY_ATTEMPTS {
        match read_config(runner, identity, backend) {
            Ok(inspection) if ready(&inspection) => return Ok(inspection),
            Ok(inspection) => last_inspection = Some(inspection),
            Err(error) => last_error = Some(error),
        }
        if attempt + 1 < CONFIG_VERIFY_ATTEMPTS {
            thread::sleep(CONFIG_VERIFY_DELAY);
        }
    }
    if let Some(inspection) = last_inspection {
        return Ok(inspection);
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Tailscale Serve status was unavailable")))
}

fn disable_mapping(runner: &impl TailscaleRunner) -> anyhow::Result<()> {
    run_checked(
        runner,
        &["serve", "--yes", "--https=443", "off"],
        "disable private Tailscale Serve",
    )?;
    Ok(())
}

fn rollback_suffix(result: &anyhow::Result<()>) -> String {
    match result {
        Ok(()) => "; the new mapping was removed".to_string(),
        Err(error) => format!(
            "; automatic removal also failed: {error}; run `tailscale serve --https=443 off` to remove the mapping manually"
        ),
    }
}

fn run_checked(
    runner: &impl TailscaleRunner,
    args: &[&str],
    operation: &str,
) -> anyhow::Result<Vec<u8>> {
    let output = runner.run(args)?;
    if output.success {
        return Ok(output.stdout);
    }
    let detail = first_line(&output.stderr)
        .or_else(|| first_line(&output.stdout))
        .unwrap_or_else(|| "no diagnostic output".to_string());
    bail!(
        "could not {operation} (tailscale exit {}): {detail}",
        output
            .code
            .map(|code| code.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    )
}

fn first_line(bytes: &[u8]) -> Option<String> {
    let line = String::from_utf8_lossy(bytes)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?
        .chars()
        .filter(|character| !character.is_control())
        .take(300)
        .collect::<String>();
    (!line.is_empty()).then_some(line)
}

fn resolve_tailscale_bin(explicit: Option<&Path>) -> anyhow::Result<PathBuf> {
    if let Some(path) = explicit {
        ensure!(
            path.is_absolute(),
            "--tailscale-bin must be an absolute path"
        );
        ensure!(
            path.is_file(),
            "Tailscale executable not found at {}",
            path.display()
        );
        return path
            .canonicalize()
            .with_context(|| format!("could not resolve {}", path.display()));
    }

    let mut candidates = Vec::new();
    #[cfg(windows)]
    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        candidates.push(
            PathBuf::from(program_files)
                .join("Tailscale")
                .join("tailscale.exe"),
        );
    }
    #[cfg(not(windows))]
    {
        candidates.push(PathBuf::from("/usr/bin/tailscale"));
        candidates.push(PathBuf::from("/usr/local/bin/tailscale"));
    }

    let executable_name = if cfg!(windows) {
        "tailscale.exe"
    } else {
        "tailscale"
    };
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(
            std::env::split_paths(&path)
                .filter(|directory| directory.is_absolute())
                .map(|directory| directory.join(executable_name)),
        );
    }

    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .and_then(|candidate| candidate.canonicalize().ok())
        .with_context(|| {
            "Tailscale CLI was not found; install Tailscale, sign in, or pass --tailscale-bin <ABSOLUTE_PATH>"
        })
}

fn run_process(
    executable: &Path,
    args: &[&str],
    timeout: Duration,
) -> anyhow::Result<CapturedOutput> {
    let mut command = Command::new(executable);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("could not start {}", executable.display()))?;
    let stdout = child
        .stdout
        .take()
        .context("could not capture Tailscale stdout")?;
    let stderr = child
        .stderr
        .take()
        .context("could not capture Tailscale stderr")?;
    let stdout_reader = thread::spawn(move || read_limited(stdout));
    let stderr_reader = thread::spawn(move || read_limited(stderr));

    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .context("could not wait for Tailscale CLI")?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            bail!(
                "Tailscale CLI did not finish within {} seconds",
                timeout.as_secs()
            );
        }
        thread::sleep(Duration::from_millis(10));
    };

    let stdout = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("Tailscale stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("Tailscale stderr reader panicked"))??;
    Ok(CapturedOutput {
        success: status.success(),
        code: status.code(),
        stdout,
        stderr,
    })
}

fn read_limited(mut reader: impl Read) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(output);
        }
        if output.len().saturating_add(read) > MAX_COMMAND_OUTPUT_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Tailscale CLI output exceeded 4 MiB",
            ));
        }
        output.extend_from_slice(&buffer[..read]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::io::Write as _;
    use std::net::TcpListener;
    use std::sync::Mutex;

    fn identity() -> TailnetIdentity {
        parse_tailnet_identity(
            br#"{"BackendState":"Running","Self":{"DNSName":"camelid-host.example.ts.net.","Online":true}}"#,
        )
        .unwrap()
    }

    fn backend() -> SocketAddr {
        DEFAULT_BACKEND.parse().unwrap()
    }

    fn health_server(api_surface: &'static str) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let read = stream.read(&mut request).unwrap();
            assert!(
                String::from_utf8_lossy(&request[..read]).starts_with("GET /v1/health HTTP/1.1")
            );
            let body = format!(r#"{{"ok":true,"api_surface":"{api_surface}"}}"#);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });
        address
    }

    fn tailscale_status() -> Vec<u8> {
        br#"{"BackendState":"Running","Self":{"DNSName":"camelid-host.example.ts.net.","Online":true}}"#.to_vec()
    }

    fn active_config() -> Vec<u8> {
        br#"{
            "TCP":{"443":{"HTTPS":true}},
            "Web":{"camelid-host.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8181"}}}}
        }"#
        .to_vec()
    }

    fn success(stdout: impl Into<Vec<u8>>) -> CapturedOutput {
        CapturedOutput {
            success: true,
            code: Some(0),
            stdout: stdout.into(),
            stderr: Vec::new(),
        }
    }

    struct FakeRunner {
        responses: Mutex<VecDeque<anyhow::Result<CapturedOutput>>>,
        calls: Mutex<Vec<Vec<String>>>,
    }

    impl FakeRunner {
        fn new(responses: Vec<anyhow::Result<CapturedOutput>>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl TailscaleRunner for FakeRunner {
        fn run(&self, args: &[&str]) -> anyhow::Result<CapturedOutput> {
            self.calls
                .lock()
                .unwrap()
                .push(args.iter().map(|arg| (*arg).to_string()).collect());
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected Tailscale call")
        }
    }

    #[test]
    fn only_the_documented_ipv4_loopback_target_is_eligible() {
        assert!(validate_backend_addr(backend()).is_ok());
        for refused in [
            "127.0.0.2:8181",
            "[::1]:8181",
            "0.0.0.0:8181",
            "192.0.2.1:8181",
        ] {
            let error = validate_backend_addr(refused.parse().unwrap()).unwrap_err();
            assert!(error.to_string().contains("127.0.0.1"), "{error:#}");
        }
    }

    #[test]
    fn the_live_backend_must_identify_itself_as_lan_chat_only() {
        assert!(require_lan_chat_backend(health_server("lan_chat_only")).is_ok());
        let error = require_lan_chat_backend(health_server("full")).unwrap_err();
        assert!(error.to_string().contains("api_surface=lan_chat_only"));
    }

    #[test]
    fn a_running_online_device_yields_its_private_https_url() {
        assert_eq!(identity().url(), "https://camelid-host.example.ts.net/");

        for raw in [
            br#"{"BackendState":"Stopped","Self":{"DNSName":"host.example.ts.net.","Online":true}}"#.as_slice(),
            br#"{"BackendState":"Running","Self":{"DNSName":"host.example.ts.net.","Online":false}}"#.as_slice(),
            br#"{"BackendState":"Running","Self":{"DNSName":"","Online":true}}"#.as_slice(),
            br#"{"BackendState":"Running","Self":{"DNSName":"host?.example.ts.net.","Online":true}}"#.as_slice(),
            br#"{"BackendState":"Running","Self":{"DNSName":"-host.example.ts.net.","Online":true}}"#.as_slice(),
        ] {
            assert!(parse_tailnet_identity(raw).is_err());
        }
    }

    #[test]
    fn tailscale_serve_cli_version_is_explicitly_bounded() {
        assert_eq!(parse_tailscale_version(b"1.52.0\n").unwrap(), (1, 52));
        assert_eq!(
            parse_tailscale_version(b"v1.94.2\nextra\n").unwrap(),
            (1, 94)
        );
        assert!(parse_tailscale_version(b"dev\n").is_err());
    }

    #[test]
    fn an_empty_config_is_startable_and_the_exact_private_mapping_is_idempotent() {
        let identity = identity();
        assert_eq!(
            inspect_serve_config(b"{}", &identity, backend()).unwrap(),
            TransportState::Inactive
        );
        assert!(require_startable(TransportState::Inactive).unwrap());

        let active = br#"{
            "TCP":{"443":{"HTTPS":true}},
            "Web":{"camelid-host.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8181"}}}}
        }"#;
        assert_eq!(
            inspect_serve_config(active, &identity, backend()).unwrap(),
            TransportState::Active
        );
        assert!(!require_startable(TransportState::Active).unwrap());
    }

    #[test]
    fn another_mapping_is_a_conflict_and_is_never_treated_as_ours() {
        let conflict = br#"{
            "TCP":{"443":{"HTTPS":true}},
            "Web":{"camelid-host.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:9999"}}}}
        }"#;
        let state = inspect_serve_config(conflict, &identity(), backend()).unwrap();
        assert_eq!(state, TransportState::Conflict);
        assert!(require_startable(state)
            .unwrap_err()
            .to_string()
            .contains("did not change"));
    }

    #[test]
    fn a_second_path_on_camelids_port_makes_the_mapping_unowned() {
        let mixed = br#"{
            "TCP":{"443":{"HTTPS":true}},
            "Web":{"camelid-host.example.ts.net:443":{"Handlers":{
                "/":{"Proxy":"http://127.0.0.1:8181"},
                "/other":{"Proxy":"http://127.0.0.1:9999"}
            }}}
        }"#;
        let state = inspect_serve_config(mixed, &identity(), backend()).unwrap();
        assert_eq!(state, TransportState::Conflict);

        let runner = FakeRunner::new(vec![
            Ok(success(tailscale_status())),
            Ok(success(mixed.to_vec())),
        ]);
        assert!(stop_transport(backend(), &runner).is_err());
        assert_eq!(runner.calls().len(), 2, "mixed mapping was mutated");
    }

    #[test]
    fn a_foreground_mapping_is_a_conflict_even_when_background_is_empty() {
        let conflict = br#"{
            "Foreground":{"session":{"TCP":{"443":{"HTTPS":true}}}}
        }"#;
        assert_eq!(
            inspect_serve_config(conflict, &identity(), backend()).unwrap(),
            TransportState::Conflict
        );
    }

    #[test]
    fn funnel_is_distinct_from_private_serve_and_always_refused() {
        let public = br#"{
            "TCP":{"443":{"HTTPS":true}},
            "Web":{"camelid-host.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8181"}}}},
            "AllowFunnel":{"camelid-host.example.ts.net:443":true}
        }"#;
        let state = inspect_serve_config(public, &identity(), backend()).unwrap();
        assert_eq!(state, TransportState::PublicFunnel);
        assert!(require_startable(state)
            .unwrap_err()
            .to_string()
            .contains("public internet"));
    }

    #[test]
    fn start_invokes_only_private_serve_and_verifies_the_result() {
        let runner = FakeRunner::new(vec![
            Ok(success(tailscale_status())),
            Ok(success(b"{}".to_vec())),
            Ok(success(Vec::new())),
            Ok(success(active_config())),
        ]);
        let status = start_transport(backend(), &runner).unwrap();
        assert_eq!(status.transport, TransportState::Active);
        assert_eq!(status.url, "https://camelid-host.example.ts.net/");
        let calls = runner.calls();
        assert_eq!(
            calls,
            vec![
                vec!["status", "--json"],
                vec!["serve", "status", "--json"],
                vec![
                    "serve",
                    "--bg",
                    "--yes",
                    "--https=443",
                    "http://127.0.0.1:8181",
                ],
                vec!["serve", "status", "--json"],
            ]
        );
        assert!(calls.iter().flatten().all(|arg| !arg.contains("funnel")));
        assert!(calls.iter().flatten().all(|arg| !arg.contains("key")));
    }

    #[test]
    fn only_serve_creation_gets_the_extended_timeout() {
        assert_eq!(
            command_timeout(&[
                "serve",
                "--bg",
                "--yes",
                "--https=443",
                "http://127.0.0.1:8181",
            ]),
            SERVE_CREATION_TIMEOUT
        );
        for args in [
            &["version"][..],
            &["status", "--json"],
            &["serve", "status", "--json"],
            &["serve", "--yes", "--https=443", "off"],
        ] {
            assert_eq!(command_timeout(args), COMMAND_TIMEOUT);
        }
    }

    #[test]
    fn start_never_mutates_a_conflicting_mapping() {
        let conflict = br#"{
            "TCP":{"443":{"HTTPS":true}},
            "Web":{"camelid-host.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:9999"}}}}
        }"#;
        let runner = FakeRunner::new(vec![
            Ok(success(tailscale_status())),
            Ok(success(conflict.to_vec())),
        ]);
        assert!(start_transport(backend(), &runner).is_err());
        assert_eq!(
            runner.calls(),
            vec![vec!["status", "--json"], vec!["serve", "status", "--json"],]
        );
    }

    #[test]
    fn failed_postcondition_removes_the_mapping_it_just_requested() {
        let runner = FakeRunner::new(vec![
            Ok(success(tailscale_status())),
            Ok(success(b"{}".to_vec())),
            Ok(success(Vec::new())),
            Ok(success(b"{}".to_vec())),
            Ok(success(b"{}".to_vec())),
            Ok(success(b"{}".to_vec())),
            Ok(success(b"{}".to_vec())),
            Ok(success(Vec::new())),
        ]);
        let error = start_transport(backend(), &runner).unwrap_err();
        assert!(error.to_string().contains("was not active"), "{error:#}");
        assert_eq!(
            runner.calls().last().unwrap(),
            &vec!["serve", "--yes", "--https=443", "off"]
        );
    }

    #[test]
    fn start_tolerates_a_delayed_serve_config_readback() {
        let runner = FakeRunner::new(vec![
            Ok(success(tailscale_status())),
            Ok(success(b"{}".to_vec())),
            Ok(success(Vec::new())),
            Ok(success(b"{}".to_vec())),
            Ok(success(active_config())),
        ]);
        assert_eq!(
            start_transport(backend(), &runner).unwrap().transport,
            TransportState::Active
        );
    }

    #[test]
    fn stop_removes_only_the_exact_camelid_root_mapping() {
        let runner = FakeRunner::new(vec![
            Ok(success(tailscale_status())),
            Ok(success(active_config())),
            Ok(success(Vec::new())),
            Ok(success(b"{}".to_vec())),
        ]);
        let status = stop_transport(backend(), &runner).unwrap();
        assert_eq!(status.transport, TransportState::Inactive);
        assert_eq!(
            runner.calls()[2],
            vec!["serve", "--yes", "--https=443", "off"]
        );
    }

    #[test]
    fn explicit_tailscale_binary_must_be_absolute_and_exist() {
        assert!(resolve_tailscale_bin(Some(Path::new("tailscale"))).is_err());
        let file = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(
            resolve_tailscale_bin(Some(file.path())).unwrap(),
            file.path().canonicalize().unwrap()
        );
    }
}
