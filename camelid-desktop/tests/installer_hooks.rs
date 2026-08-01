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

const SIGN_SCRIPT: &str = "windows/sign-artifact-signing.ps1";

fn release_workflow() -> String {
    let path = desktop_dir()
        .parent()
        .expect("repo root")
        .join(".github/workflows/release.yml");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// `bundle.windows.signCommand` is the only hook that can sign the copy of
/// `camelid-desktop.exe` sealed inside the NSIS installer: Tauri patches the binary with the
/// bundle type and signs only afterwards, so a signature applied before `tauri build` is
/// invalidated and one applied after cannot reach a binary already inside the installer.
/// v0.4.6 shipped that copy `NotSigned`; v0.4.7 shipped it `HashMismatch`.
///
/// It is wired by a config the RELEASE WORKFLOW generates, not by the committed
/// `tauri.conf.json`, because the paths must be absolute and machine-specific — see
/// `sign_command_paths_must_be_absolute`.
#[test]
fn release_workflow_generates_the_signing_config_and_passes_it_to_the_bundler() {
    let wf = release_workflow();

    assert!(
        wf.contains("camelid-desktop/tauri.signing.conf.json"),
        "the release workflow must generate tauri.signing.conf.json; without it nothing signs \
         the binary inside the installer"
    );
    assert!(
        wf.contains("--config tauri.bundle.conf.json --config tauri.signing.conf.json"),
        "the bundler must be passed the generated signing config, or signCommand never runs"
    );
    assert!(
        wf.contains(SIGN_SCRIPT),
        "the generated signing config must point at {SIGN_SCRIPT}"
    );
    assert!(
        desktop_dir().join(SIGN_SCRIPT).is_file(),
        "{SIGN_SCRIPT} is referenced by the release workflow but does not exist"
    );
}

/// Both paths in the generated signCommand must be ABSOLUTE.
///
/// Tauri invokes signCommand seven times per bundle — the app binary, five NSIS plugin DLLs,
/// and the uninstaller staged as a `%TEMP%\nst*.tmp` — and the working directory is not
/// constant: the uninstaller call runs from `target\release\nsis\x64`. v0.4.8 wired a
/// project-relative script path, died there with `failed to run powershell`, and published no
/// Windows installer at all. A bare `powershell` also failed to spawn on the runner, so the
/// interpreter is resolved to a full path too.
#[test]
fn sign_command_paths_must_be_absolute() {
    let wf = release_workflow();

    assert!(
        wf.contains("Resolve-Path 'camelid-desktop/windows/sign-artifact-signing.ps1'"),
        "the signing script path must be resolved to an absolute path before it reaches \
         signCommand: one of Tauri's seven invocations runs from target\\release\\nsis\\x64, \
         where a project-relative path does not resolve"
    );
    assert!(
        wf.contains("(Get-Command powershell.exe).Source"),
        "the interpreter must be resolved to a full path: a bare `powershell` failed to spawn \
         on the release runner"
    );
    assert!(
        wf.contains("installerHooks = $hooks"),
        "the generated signing config must carry installerHooks through, so it cannot silently \
         drop the NVRTC-orphan cleanup if Tauri's config merge is not deep at bundle.windows"
    );
}

/// The signing script's failure policy, which is deliberately asymmetric.
///
/// A signtool failure must NOT fail the bundle: the release still has to produce an installer.
/// Failing there is what left users with no Windows installer at all on v0.4.8 and v0.5.0 —
/// strictly worse than the unsigned-payload installer v0.4.6 shipped, which installed and ran.
///
/// A signature that does not verify IS fatal: v0.4.7 shipped `HashMismatch` on every installed
/// copy, and Windows reports that as a tampered chain. Broken is worse than absent.
#[test]
fn sign_script_ships_unsigned_on_failure_but_never_ships_a_broken_signature() {
    let src =
        std::fs::read_to_string(desktop_dir().join(SIGN_SCRIPT)).expect("read signing script");

    assert!(
        src.contains("$signExit -ne 0"),
        "the signing script must capture and check signtool's exit code"
    );
    assert!(
        src.contains("shipping this file UNSIGNED rather than failing the bundle"),
        "a signtool failure must be non-fatal, or a signing outage means users get no installer"
    );
    assert!(
        src.contains("refusing to seal a broken signature"),
        "a result that does not verify must be fatal: a broken signature is worse than none"
    );
    assert!(
        src.contains("-ne 'Valid'"),
        "the script must verify the resulting signature, not just signtool's exit code"
    );
}

/// The hook must log to a file, because Tauri DISCARDS its output on a non-zero exit and
/// reports only `failed to bundle project: failed to run <cmd>`. Three consecutive releases
/// failed here with nothing else to go on, and each diagnosis was guesswork.
///
/// It must also drop `$ErrorActionPreference` for the native signtool call: Windows PowerShell
/// wraps a redirected native command's stderr in NativeCommandError records, and under `Stop`
/// that is terminating — the script would die on signtool's first diagnostic line and never
/// reach its own exit-code handling, turning any hiccup into an unexplained bundle failure.
#[test]
fn sign_script_is_observable_when_tauri_swallows_its_output() {
    let src =
        std::fs::read_to_string(desktop_dir().join(SIGN_SCRIPT)).expect("read signing script");
    let wf = release_workflow();

    assert!(
        src.contains("CAMELID_SIGN_LOG"),
        "the signing script must write to a log file; Tauri discards its stdout on failure"
    );
    assert!(
        src.contains("$ErrorActionPreference = 'Continue'"),
        "the native signtool call must run with ErrorActionPreference dropped, or a stderr line \
         terminates the script before its exit-code handling runs"
    );
    assert!(
        wf.contains("CAMELID_SIGN_LOG=$signLog"),
        "the release workflow must set CAMELID_SIGN_LOG"
    );
    assert!(
        wf.contains("name: Sign-command log"),
        "the release workflow must print the sign log so a failure is diagnosable from CI alone"
    );
    assert!(
        src.contains("trap {"),
        "the script needs a trap: v0.5.1 signed successfully and then exited non-zero with the \
         log ending mid-script, because an unhandled terminating error names itself only on \
         stderr — which Tauri discards"
    );
}

/// Reading the signature back must never be able to kill the bundle.
///
/// signtool has just closed the file and the signing service wrote to it moments earlier, so
/// the read can lose a race against antivirus or a lingering handle. On v0.5.1 signtool reported
/// `Number of errors: 0` and the script still exited non-zero 0.28s later, in exactly this read.
///
/// Unreadable is not the same as broken: failing there discards a correctly signed binary over a
/// transient lock and lands back on the no-installer outcome. A CONFIRMED non-`Valid` result
/// stays fatal.
#[test]
fn signature_readback_retries_and_an_unreadable_result_is_not_fatal() {
    let src =
        std::fs::read_to_string(desktop_dir().join(SIGN_SCRIPT)).expect("read signing script");

    assert!(
        src.contains("for ($attempt = 1; $attempt -le 5; $attempt++)"),
        "the signature read-back must retry; it races the signing service's own write"
    );
    assert!(
        src.contains("could not read back the signature"),
        "an unreadable signature must warn and continue, not fail the bundle"
    );
    assert!(
        src.contains("refusing to seal a broken signature"),
        "a CONFIRMED non-Valid signature must still be fatal"
    );
}

/// An incomplete release must never become `latest`. The desktop jobs are independent by
/// design, so a desktop failure publishes the server artifacts anyway — which is what left
/// v0.4.8 as `latest` with no Windows installer and broke the documented one-command install
/// for every user. The guard demotes such a release to a prerelease so `releases/latest` skips
/// it and the previous complete release keeps serving installs.
#[test]
fn release_is_demoted_when_its_asset_set_is_incomplete() {
    let wf = release_workflow();

    assert!(
        wf.contains("verify-release-assets"),
        "the release workflow must verify the published asset set"
    );
    assert!(
        wf.contains("--prerelease --latest=false"),
        "an incomplete release must be demoted so releases/latest cannot point at it"
    );
    assert!(
        wf.contains("x64-setup\\.exe$"),
        "the completeness check must require the NSIS installer — that is the asset \
         scripts/get-desktop-windows.ps1 resolves"
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
