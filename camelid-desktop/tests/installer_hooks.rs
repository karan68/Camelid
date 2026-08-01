//! The Windows NSIS installer-hook contract (see `../DECISIONS.md`, D11 cont.).
//!
//! An overwrite-only installer rewrites the files it ships and leaves everything else alone, so
//! a file installed by an OLDER version that the current one dropped survives every upgrade —
//! that is how an 85.7 MB `nvrtc64_120_0.alt.dll` stranded on v0.4.6 boxes. `windows/installer-hooks.nsh`
//! closes that path by re-laying the NVRTC set on every install.
//!
//! These tests exist because nothing else can see this. The hook is NSIS source, invisible to
//! the compiler; `scripts/check-release-artifact.mjs` inspects the built artifact, not the
//! upgrade path; and the failure is silent — a stranded file changes no behavior, it just
//! accumulates. The wholesale-clear guard below is the load-bearing one: `sidecar\models\` is
//! the desktop's model store, so widening the hook is a data-loss bug, not a cleanup.

use std::path::PathBuf;

fn desktop_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The hook path as the NSIS bundler resolves it: `bundle.windows.nsis.installerHooks`,
/// relative to the Tauri project directory.
fn hooks_path() -> PathBuf {
    let raw = std::fs::read_to_string(desktop_dir().join("tauri.conf.json"))
        .expect("read tauri.conf.json");
    let conf: serde_json::Value =
        serde_json::from_str(&raw).expect("tauri.conf.json is valid JSON");
    let rel = conf["bundle"]["windows"]["nsis"]["installerHooks"]
        .as_str()
        .expect(
            "bundle.windows.nsis.installerHooks must stay configured: without it an in-place \
             upgrade silently strands every sidecar file the current version no longer ships",
        );
    desktop_dir().join(rel)
}

fn hooks_source() -> String {
    let path = hooks_path();
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read installer hooks at {}: {e}", path.display()))
}

/// NSIS code on a line, with any trailing `;` or `#` comment removed.
fn code_of(line: &str) -> &str {
    let end = line.find([';', '#']).unwrap_or(line.len());
    line[..end].trim()
}

#[test]
fn installer_hooks_file_is_wired_and_present() {
    let path = hooks_path();
    assert!(
        path.is_file(),
        "installerHooks points at {}, which does not exist — `tauri build` fails at release time",
        path.display()
    );
}

#[test]
fn preinstall_hook_relays_the_whole_nvrtc_set() {
    let src = hooks_source();
    assert!(
        src.contains("!macro NSIS_HOOK_PREINSTALL"),
        "the cleanup must run in NSIS_HOOK_PREINSTALL, before the file copy re-lays the set"
    );
    // Both families, because both encode the CUDA version in the filename: a CUDA_VERSION bump
    // renames them, and a rename is exactly what an overwrite-only installer cannot fix.
    for pattern in [
        r#"Delete "$INSTDIR\sidecar\nvrtc64_*.dll""#,
        r#"Delete "$INSTDIR\sidecar\nvrtc-builtins64_*.dll""#,
    ] {
        assert!(
            src.contains(pattern),
            "installer hook must contain `{pattern}` so superseded NVRTC files cannot strand"
        );
    }
}

#[test]
fn hook_never_clears_the_sidecar_directory_wholesale() {
    // `sidecar_models_dir` (src/engine.rs) puts the desktop's model store at
    // $INSTDIR\sidecar\models\ — multi-GB user-downloaded GGUF weights. Clearing sidecar\ to
    // catch orphans would delete them. Every removal here must name an explicit NVRTC file.
    for line in hooks_source().lines() {
        let code = code_of(line);
        if code.is_empty() {
            continue;
        }
        let lower = code.to_ascii_lowercase();
        assert!(
            !lower.starts_with("rmdir"),
            "RMDir in an installer hook can take sidecar\\models\\ (user model weights) with it: {code}"
        );
        if lower.starts_with("delete") {
            assert!(
                code.contains(r"\sidecar\nvrtc"),
                "every Delete must name an explicit sidecar NVRTC file, never a broader glob: {code}"
            );
        }
    }
}

#[test]
fn engine_still_resolves_models_beside_the_sidecar_binary() {
    // The guard above is only meaningful while this stays true. If the model store ever moves
    // out of $INSTDIR, this test fails and the wholesale-clear reasoning must be revisited
    // rather than silently inherited.
    let engine = std::fs::read_to_string(desktop_dir().join("src").join("engine.rs"))
        .expect("read src/engine.rs");
    assert!(
        engine.contains(r#"parent.join("models")"#),
        "sidecar_models_dir no longer resolves models/ beside the engine binary — re-check \
         whether $INSTDIR\\sidecar\\ still holds user data before widening the installer hook"
    );
}
