//! The Windows bundler-hook contract (see `../DECISIONS.md`, D11 cont.) — two hooks that the
//! compiler cannot see and that fail silently, so they are asserted here instead.
//!
//! **`installerHooks`** — an overwrite-only installer rewrites the files it ships and leaves
//! everything else alone, so a file installed by an OLDER version that the current one dropped
//! survives every upgrade; that is how an 85.7 MB `nvrtc64_120_0.alt.dll` stranded on v0.4.6
//! boxes. `windows/installer-hooks.nsh` closes that path by re-laying the NVRTC set on every
//! install. The wholesale-clear guard below is the load-bearing one: `sidecar\models\` is the
//! desktop's model store, so widening the hook is a data-loss bug, not a cleanup.
//!
//! **`signCommand`** — Tauri rewrites the bundle-type marker on the copy of the binary it
//! stages for the installer, so that copy can only be signed from inside the bundler. Losing
//! this config does not fail the build; it ships an installer whose payload Windows refuses to
//! trust. v0.4.6 shipped it `NotSigned`, v0.4.7 `HashMismatch`.
//!
//! Neither failure is visible to anything else in the tree: both hooks are non-Rust source,
//! and `scripts/check-release-artifact.mjs` inspects the built artifact rather than the
//! installer's payload or the upgrade path. The release workflow's `Prove the INSTALLED binary
//! is signed` step is the runtime counterpart to these compile-time assertions.

use std::path::PathBuf;

fn desktop_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn tauri_conf() -> serde_json::Value {
    let raw = std::fs::read_to_string(desktop_dir().join("tauri.conf.json"))
        .expect("read tauri.conf.json");
    serde_json::from_str(&raw).expect("tauri.conf.json is valid JSON")
}

/// `bundle.windows.signCommand` is the ONLY hook that can sign the copy of
/// `camelid-desktop.exe` sealed inside the NSIS installer. Tauri rewrites the bundle-type
/// marker (`__TAURI_BUNDLE_TYPE_VAR_UNK` -> `..._NSS`) on a staged copy, so a signature
/// applied before `tauri build` is invalidated by that rewrite, and one applied afterwards
/// cannot reach a binary already inside the installer. v0.4.6 shipped that copy `NotSigned`;
/// v0.4.7 shipped it `HashMismatch`. Both are what dropping this config looks like.
#[test]
fn sign_command_is_wired_to_a_script_that_exists() {
    let conf = tauri_conf();
    let sign = &conf["bundle"]["windows"]["signCommand"];
    assert!(
        !sign.is_null(),
        "bundle.windows.signCommand must stay configured: it is the only point at which the \
         binary inside the NSIS installer can be signed"
    );

    let args: Vec<&str> = sign["args"]
        .as_array()
        .expect("signCommand must use the object notation so paths may contain whitespace")
        .iter()
        .map(|a| a.as_str().expect("signCommand args are strings"))
        .collect();
    assert!(
        args.iter().any(|a| *a == "%1"),
        "signCommand args must contain the %1 placeholder, or Tauri passes no file to sign: {args:?}"
    );

    let script = args
        .iter()
        .find(|a| a.ends_with(".ps1"))
        .expect("signCommand must invoke a .ps1 script");
    let path = desktop_dir().join(script);
    assert!(
        path.is_file(),
        "signCommand points at {}, which does not exist — the release bundler would fail",
        path.display()
    );
}

/// The signing script must fail rather than return success when it cannot sign. A hook that
/// swallows an error hands the bundler an unsigned binary and reports nothing, which is
/// exactly the silent failure the release gate exists to catch.
#[test]
fn sign_script_fails_closed_on_a_signing_error() {
    let conf = tauri_conf();
    let args = conf["bundle"]["windows"]["signCommand"]["args"]
        .as_array()
        .expect("signCommand args");
    let script = args
        .iter()
        .filter_map(|a| a.as_str())
        .find(|a| a.ends_with(".ps1"))
        .expect("signCommand script");
    let src = std::fs::read_to_string(desktop_dir().join(script)).expect("read signing script");

    assert!(
        src.contains("$LASTEXITCODE -ne 0"),
        "the signing script must check signtool's exit code"
    );
    assert!(
        src.contains("-ne 'Valid'"),
        "the signing script must verify the resulting signature, not just signtool's exit code"
    );
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
