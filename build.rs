use std::{env, path::PathBuf, process::Command};

fn main() {
    embed_build_provenance();
    println!("cargo:rerun-if-changed=src/x86_amx_q8.c");
    println!("cargo:rerun-if-env-changed=CAMELID_BUILD_X86_AMX_SHIM");
    println!("cargo:rustc-check-cfg=cfg(camelid_x86_amx_shim)");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        build_dg_argsort_shim();
    }
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os != "linux" || target_arch != "x86_64" {
        return;
    }
    let require_amx_shim = env_flag_enabled("CAMELID_BUILD_X86_AMX_SHIM");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let obj = out_dir.join("x86_amx_q8.o");
    let lib = out_dir.join("libcamelid_x86_amx_q8.a");

    let status = Command::new("gcc")
        .args([
            "-O3",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-mavx512f",
            "-mfma",
            "-mamx-tile",
            "-mamx-int8",
            "-c",
            "src/x86_amx_q8.c",
            "-o",
        ])
        .arg(&obj)
        .status();
    let Ok(status) = status else {
        if require_amx_shim {
            panic!("failed to run gcc for x86 AMX Q8 kernel");
        }
        println!("cargo:warning=skipping optional x86 AMX Q8 shim because gcc could not be run");
        return;
    };
    if !status.success() {
        if require_amx_shim {
            panic!("gcc failed building x86 AMX Q8 kernel");
        }
        println!(
            "cargo:warning=skipping optional x86 AMX Q8 shim because gcc rejected the AMX flags"
        );
        return;
    }

    let status = Command::new("ar").arg("crus").arg(&lib).arg(&obj).status();
    let Ok(status) = status else {
        if require_amx_shim {
            panic!("failed to run ar for x86 AMX Q8 kernel");
        }
        println!("cargo:warning=skipping optional x86 AMX Q8 shim because ar could not be run");
        return;
    };
    if !status.success() {
        if require_amx_shim {
            panic!("ar failed building x86 AMX Q8 kernel");
        }
        println!("cargo:warning=skipping optional x86 AMX Q8 shim because ar failed");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=camelid_x86_amx_q8");
    println!("cargo:rustc-cfg=camelid_x86_amx_shim");
}

// DiffusionGemma expert-argsort shim: compile src/dg_argsort.cpp with the
// system C++ toolchain (Apple clang libc++ — identical to the pinned reference
// build) so camelid's MoE expert ordering matches the reference's libc++
// std::sort tie behavior bit-for-bit. macOS only; the diffusion-gemma lane is
// Apple-Silicon-only.
fn build_dg_argsort_shim() {
    println!("cargo:rerun-if-changed=src/dg_argsort.cpp");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let obj = out_dir.join("dg_argsort.o");
    let lib = out_dir.join("libdg_argsort.a");
    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let status = Command::new(&cxx)
        .args(["-std=c++17", "-O2", "-c", "src/dg_argsort.cpp", "-o"])
        .arg(&obj)
        .status()
        .expect("failed to run c++ for dg_argsort shim");
    assert!(status.success(), "c++ failed building dg_argsort shim");
    let status = Command::new("ar")
        .arg("crus")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("failed to run ar for dg_argsort shim");
    assert!(status.success(), "ar failed building dg_argsort shim");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=dg_argsort");
    // libc++ for the std::sort instantiation
    println!("cargo:rustc-link-lib=c++");
}

// Embed git provenance so a running binary reports its own version/commit
// (used by parity receipts) without shelling out at request time. Builds
// without a git checkout simply omit the env vars; the receipt module falls
// back to the crate version.
fn embed_build_provenance() {
    // Re-run when HEAD or the index moves so the embedded commit stays current.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    if let Some(commit) = git_stdout(&["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=CAMELID_GIT_COMMIT={commit}");
    }
    if let Some(describe) = git_stdout(&["describe", "--tags", "--dirty"]) {
        println!("cargo:rustc-env=CAMELID_GIT_DESCRIBE={describe}");
    }
}

fn git_stdout(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn env_flag_enabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}
