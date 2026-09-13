//! Durable project settings and bounded, evidence-producing verification.
//! These records carry observations, never command approval authority.
use super::tools::Sandbox;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ProjectSettings {
    pub engine_id: String,
    pub engine_name: String,
    pub verification_command: String,
    pub preview_entry: String,
    pub workflow: String,
    pub max_run_seconds: u64,
    pub browser_check: bool,
}
impl ProjectSettings {
    pub fn validate(&self) -> Result<(), String> {
        if self.verification_command.len() > 4000
            || self.workflow.len() > 80
            || self.engine_id.len() > 64
            || self.engine_name.len() > 200
            || (self.max_run_seconds != 0 && !(60..=7200).contains(&self.max_run_seconds))
        {
            return Err(
                "Project settings exceed their limits (run budget: 60–7200 seconds).".into(),
            );
        }
        if !self.preview_entry.is_empty() {
            relative_path(&self.preview_entry)?;
        }
        Ok(())
    }
    pub fn budget(&self) -> u64 {
        if self.max_run_seconds == 0 {
            1800
        } else {
            self.max_run_seconds
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Check {
    pub id: String,
    pub run_id: String,
    pub revision: u64,
    pub time: u64,
    pub command: String,
    pub status: String,
    pub output: String,
    pub files: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Workflow {
    pub name: String,
    pub notes: String,
    pub verification_command: String,
    pub preview_entry: String,
    pub source_check: String,
    #[serde(default)]
    pub browser_check: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Checkpoint {
    pub id: String,
    pub run_id: String,
    pub title: String,
    pub review_ids: Vec<String>,
    pub status: String,
}
pub(crate) struct PreparedCheck {
    pub command: String,
    pub files: BTreeMap<String, String>,
    pub browser: bool,
    pub _temporary: Option<tempfile::TempDir>,
}

pub(crate) fn prepare_check(
    root: &Path,
    settings: &ProjectSettings,
    command: String,
    files: BTreeMap<String, String>,
) -> Result<PreparedCheck, String> {
    if !settings.browser_check {
        return Ok(PreparedCheck {
            command,
            files,
            browser: false,
            _temporary: None,
        });
    }
    if cfg!(windows) {
        return Err("Built-in browser load checks currently require a Unix engine. Use a project browser-test command on this engine.".into());
    }
    let chrome = std::env::var_os("CAMELID_BROWSER_PATH").map(PathBuf::from).or_else(|| {
        ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"].into_iter().map(PathBuf::from).find(|p| p.is_file())
    }).ok_or("No Chromium browser found on this engine. Configure CAMELID_BROWSER_PATH or disable the browser load check and use a project test command.")?;
    let chrome = fs::canonicalize(chrome).map_err(|e| e.to_string())?;
    if !chrome.is_absolute() || !chrome.is_file() {
        return Err("Browser executable is unavailable.".into());
    }
    let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
    let page = directory.path().join("check.html");
    let html = preview(root, &settings.preview_entry)?;
    let probe = r#"<script>(()=>{const errors=[];addEventListener('error',e=>errors.push(e.message||'A page resource failed to load'),true);addEventListener('unhandledrejection',e=>errors.push(String(e.reason)));addEventListener('load',()=>setTimeout(()=>{const result={kind:'browser_load',passed:errors.length===0,errors,scope:'Page load and JavaScript errors only; interactions were not tested'};document.documentElement.innerHTML='<head></head><body><pre id="camelid-check-result">'+btoa(unescape(encodeURIComponent(JSON.stringify(result))))+'</pre></body>';},150));})();</script>"#;
    fs::write(&page, format!("{probe}{html}")).map_err(|e| e.to_string())?;
    let quote = |s: &str| format!("'{}'", s.replace('\'', "'\\''"));
    let runner = directory.path().join("browser-check.cjs");
    let args = serde_json::json!([
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        format!(
            "--user-data-dir={}",
            directory.path().join("profile").display()
        ),
        "--virtual-time-budget=2000",
        "--dump-dom",
        format!("file://{}", page.display())
    ]);
    // Some headless macOS Chrome builds print their DOM but leave the browser
    // process alive. Own completion explicitly. Chrome stays in the approved
    // shell's process group, which run_shell tears down on exit/cancel/timeout.
    let runner_source = format!(
        r#"const {{spawn}}=require('node:child_process');
const child=spawn({}, {}, {{stdio:['ignore','pipe','ignore']}});
let output='',finished=false;
const end=(code,message)=>{{if(finished)return;finished=true;clearTimeout(timer);child.kill('SIGTERM');process.stdout.write(message+'\n',()=>process.exit(code));}};
const timer=setTimeout(()=>end(1,'Browser did not produce a page-load report within 45 seconds.'),45000);
child.stdout.on('data',chunk=>{{output+=chunk.toString();const match=output.match(/<pre id="camelid-check-result">[^<]+<\/pre>/);if(match)end(0,match[0]);else if(output.length>262144)end(1,'Browser report exceeded its output limit.');}});
child.on('error',error=>end(1,'Browser launch failed: '+error.message));
child.on('close',()=>{{if(!finished)end(1,'Browser exited without a page-load report.');}});
"#,
        serde_json::to_string(&chrome.to_string_lossy()).map_err(|e| e.to_string())?,
        args
    );
    fs::write(&runner, runner_source).map_err(|e| e.to_string())?;
    let browser = format!("node {}", quote(&runner.to_string_lossy()));
    let command = if command.trim().is_empty() {
        browser
    } else {
        format!("({command}) && {browser}")
    };
    Ok(PreparedCheck {
        command,
        files,
        browser: true,
        _temporary: Some(directory),
    })
}
pub(crate) fn browser_result(output: &str) -> Result<String, String> {
    use base64::Engine;
    let marker = "<pre id=\"camelid-check-result\">";
    let encoded = output
        .split_once(marker)
        .and_then(|(_, tail)| tail.split_once("</pre>").map(|(text, _)| text))
        .ok_or(
            "The browser did not return a verification report; no page-load pass was recorded.",
        )?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|_| "Invalid browser report.")?;
    let report: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| "Invalid browser report JSON.")?;
    if report["passed"] == true {
        Ok(report.to_string())
    } else {
        Err(report.to_string())
    }
}

pub(crate) fn relative_path(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 1024
        || value.contains(['\\', ':', '\0', '?', '#'])
        || Path::new(value)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        || value
            .split('/')
            .any(|s| s.eq_ignore_ascii_case(".git") || s.eq_ignore_ascii_case(".camelid"))
    {
        return Err(
            "Use a relative project file path without hidden directories or parent traversal."
                .into(),
        );
    }
    Ok(())
}
pub(crate) fn read_file(root: &Path, relative: &str) -> Result<Vec<u8>, String> {
    relative_path(relative)?;
    let sandbox = Sandbox::new(root, false, Duration::from_secs(1)).map_err(|e| e.to_string())?;
    if sandbox.root() != root {
        return Err("The project folder changed.".into());
    }
    let mut path = root.to_path_buf();
    for c in Path::new(relative).components() {
        path.push(c);
        if fs::symlink_metadata(&path)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("Project evidence cannot follow symbolic links.".into());
        }
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|e| e.to_string())?;
    if !file
        .metadata()
        .is_ok_and(|m| m.is_file() && m.len() <= 256 * 1024)
    {
        return Err("Preview and check evidence require regular files of at most 256 KB.".into());
    }
    let mut bytes = vec![];
    file.take(256 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 256 * 1024 {
        return Err("Project file exceeds its size limit.".into());
    }
    Ok(bytes)
}
pub(crate) fn fingerprints(
    root: &Path,
    paths: impl Iterator<Item = String>,
) -> Result<BTreeMap<String, String>, String> {
    let mut result = BTreeMap::new();
    for path in paths {
        if result.len() >= 128 {
            return Err("Too many files in check evidence.".into());
        }
        let bytes = read_file(root, &path)?;
        result.insert(path, format!("{:x}", Sha256::digest(bytes)));
    }
    Ok(result)
}

pub(crate) fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let parent = path.parent().ok_or("Missing store directory.")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    if fs::symlink_metadata(parent)
        .map_err(|e| e.to_string())?
        .file_type()
        .is_symlink()
    {
        return Err("Project store cannot be a symbolic link.".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
    }
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    file.write_all(&serde_json::to_vec(value).map_err(|e| e.to_string())?)
        .and_then(|_| file.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    file.persist(path).map_err(|e| e.to_string())?;
    Ok(())
}
pub(crate) fn engine(directory: &Path) -> Result<(String, String), String> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().map_err(|_| "Engine identity unavailable.")?;
    let host = host_name();
    let name = std::env::var("CAMELID_EXECUTION_NAME")
        .ok()
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| host.clone())
        .trim()
        .chars()
        .take(200)
        .collect();
    let path = directory.join("execution-engine.json");
    let key: String = if path.exists() {
        let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 1024 {
            return Err("Invalid engine identity.".into());
        }
        serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        let key = uuid::Uuid::new_v4().simple().to_string();
        atomic_json(&path, &key)?;
        key
    };
    if key.len() != 32 || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Invalid engine identity.".into());
    }
    // A copied session store does not silently rebind to another host.
    let bound = format!("{:x}", Sha256::digest(format!("{key}:{host}")));
    Ok((bound[..32].into(), name))
}
fn host_name() -> String {
    #[cfg(unix)]
    {
        let mut name = [0u8; 256];
        // gethostname writes at most this fixed buffer; force NUL termination.
        if unsafe { libc::gethostname(name.as_mut_ptr().cast(), name.len() - 1) } == 0 {
            return String::from_utf8_lossy(
                &name[..name.iter().position(|b| *b == 0).unwrap_or(255)],
            )
            .into_owned();
        }
    }
    std::env::var("COMPUTERNAME").unwrap_or_else(|_| "Connected engine".into())
}
pub(crate) fn workflow_file(
    directory: &Path,
    workspace: &Path,
    name: &str,
) -> Result<PathBuf, String> {
    if name.is_empty()
        || name.len() > 80
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(
            "Workflow names use letters, numbers, hyphens, or underscores (1–80 characters)."
                .into(),
        );
    }
    Ok(directory
        .join("workflows")
        .join(format!(
            "{:x}",
            Sha256::digest(workspace.to_string_lossy().as_bytes())
        ))
        .join(format!("{name}.json")))
}
pub(crate) fn read_workflow(
    directory: &Path,
    workspace: &Path,
    name: &str,
) -> Result<Workflow, String> {
    let path = workflow_file(directory, workspace, name)?;
    let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 32 * 1024 {
        return Err("Invalid saved workflow.".into());
    }
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// A single-file, script-capable preview runs in an opaque iframe origin. Local
/// assets are inlined; its scripts receive no access to the engine or chat DOM.
/// Module/dev-server applications should configure their own verification command.
pub(crate) fn preview(root: &Path, entry: &str) -> Result<String, String> {
    let entry = if entry.is_empty() {
        "index.html"
    } else {
        entry
    };
    let mut html = String::from_utf8(read_file(root, entry)?)
        .map_err(|_| "The preview entry must be UTF-8 HTML.")?;
    if !entry.ends_with(".html") {
        return Err("Choose an HTML preview entry.".into());
    }
    let base = Path::new(entry).parent().unwrap_or(Path::new(""));
    // Handle quoted local script/link attributes without interpreting model text
    // as a filesystem path. Unsupported assets remain visible as load failures.
    for (tag, attribute, ending) in [("script", "src", "</script>"), ("link", "href", "")] {
        let mut from = 0;
        while let Some(offset) = html[from..].find(&format!("<{tag}")) {
            let start = from + offset;
            let Some(end_offset) = html[start..].find('>') else {
                break;
            };
            let end = start + end_offset + 1;
            let opening = &html[start..end];
            let source = [format!("{attribute}=\""), format!("{attribute}='")]
                .iter()
                .find_map(|prefix| {
                    let i = opening.find(prefix)? + prefix.len();
                    let quote = prefix.chars().last()?;
                    Some(opening[i..].split(quote).next()?.to_string())
                });
            let Some(source) = source else {
                from = end;
                continue;
            };
            if tag == "link" && !opening.contains("stylesheet") {
                from = end;
                continue;
            }
            let relative = base
                .join(source.strip_prefix("./").unwrap_or(&source))
                .to_string_lossy()
                .replace('\\', "/");
            let bytes = read_file(root, &relative)?;
            let content = String::from_utf8(bytes).map_err(|_| "Preview assets must be UTF-8.")?;
            let replacement = if tag == "script" {
                format!(
                    "<script>{}</script>",
                    content.replace("</script", "<\\/script")
                )
            } else {
                format!("<style>{}</style>", content.replace("</style", "<\\/style"))
            };
            let end = if ending.is_empty() {
                end
            } else {
                end + html[end..]
                    .find(ending)
                    .ok_or("Missing closing script tag.")?
                    + ending.len()
            };
            html.replace_range(start..end, &replacement);
            from = start + replacement.len();
            if html.len() > 2 * 1024 * 1024 {
                return Err("Preview exceeds 2 MB.".into());
            }
        }
    }
    // CSP also applies if this HTML is accidentally displayed outside the iframe.
    let policy = "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'\">";
    Ok(format!("{policy}{html}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preview_inlines_local_assets_and_rejects_escaping_sources() {
        let temp = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(temp.path()).unwrap();
        fs::write(root.join("index.html"), "<html><head><link rel=\"stylesheet\" href=\"style.css\"></head><body><script src=\"app.js\"></script></body></html>").unwrap();
        fs::write(root.join("style.css"), "body { color: red }").unwrap();
        fs::write(root.join("app.js"), "document.body.dataset.loaded='yes'").unwrap();
        let html = preview(&root, "index.html").unwrap();
        assert!(html.contains("color: red"));
        assert!(html.contains("dataset.loaded"));
        assert!(html.contains("connect-src 'none'"));
        fs::write(
            root.join("index.html"),
            "<script src='../secret.js'></script>",
        )
        .unwrap();
        assert!(preview(&root, "index.html").is_err());
        assert!(preview(&root, ".git/config").is_err());
    }
    #[test]
    fn workflow_storage_is_project_scoped_and_browser_reports_require_evidence() {
        let temp = tempfile::tempdir().unwrap();
        assert_ne!(
            workflow_file(temp.path(), Path::new("/one"), "build").unwrap(),
            workflow_file(temp.path(), Path::new("/two"), "build").unwrap()
        );
        assert!(workflow_file(temp.path(), Path::new("/one"), "../escape").is_err());
        assert!(browser_result("exit: 0\nOpened index.html").is_err());
        use base64::Engine;
        for passed in [true, false] {
            let encoded = base64::engine::general_purpose::STANDARD
                .encode(serde_json::json!({"passed":passed,"scope":"load only"}).to_string());
            assert_eq!(
                browser_result(&format!("<pre id=\"camelid-check-result\">{encoded}</pre>"))
                    .is_ok(),
                passed
            );
        }
    }
    #[cfg(unix)]
    #[test]
    #[ignore = "Launches a real Chromium browser; run on the designated test host only"]
    fn chromium_load_check_distinguishes_valid_and_broken_javascript() {
        use super::super::{shell_sandbox::ShellSandbox, tools::Action};
        for broken in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let root = fs::canonicalize(temp.path()).unwrap();
            fs::write(root.join("index.html"), "<!doctype html><html><body><h1>Fixture</h1><script src='app.js'></script></body></html>").unwrap();
            fs::write(
                root.join("app.js"),
                if broken {
                    "const broken = ;"
                } else {
                    "document.body.dataset.ready='yes';"
                },
            )
            .unwrap();
            let settings = ProjectSettings {
                browser_check: true,
                preview_entry: "index.html".into(),
                ..Default::default()
            };
            let prepared = prepare_check(&root, &settings, String::new(), BTreeMap::new()).unwrap();
            let sandbox = Sandbox::new(&root, false, Duration::from_secs(30))
                .unwrap()
                .with_shell_mode(ShellSandbox::Unrestricted);
            let result = Action::RunShell {
                command: prepared.command.clone(),
            }
            .execute_with_cancel(&sandbox, &std::sync::atomic::AtomicBool::new(false));
            assert!(!result.is_err(), "{}", result.text());
            assert_eq!(
                browser_result(result.text()).is_err(),
                broken,
                "{}",
                result.text()
            );
        }
    }
}
