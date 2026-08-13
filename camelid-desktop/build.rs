fn main() {
    // The app's own commands must be DECLARED here for tauri-build to generate
    // their `allow-*` / `deny-*` ACL permissions, and every command the app
    // actually invokes must ALSO be granted by a capability.
    //
    // The local splash is NOT exempt. Declaring any command here makes
    // `Resolved::has_app_acl` true, and tauri's invoke gate
    // (`plugin_command.is_some() || has_app_acl_manifest || !is_local`) then
    // enforces the ACL for local origins too — so an undeclared command is
    // rejected with "not allowed by ACL" even when the splash calls it.
    // `startup_snapshot` is the proof: it is called only by the local splash
    // and still needs `allow-startup-snapshot` in capabilities/default.json.
    //
    // Miss either half and the failure is SILENT — it compiles, links, and
    // passes CI, then rejects at runtime. `retry_startup` shipped that way:
    // registered in `generate_handler!` and wired to the splash's Retry
    // button, but declared in neither place, so Retry hid the error pane and
    // then died in the ACL, stranding the user on a permanent spinner with the
    // diagnostics already erased.
    let attributes =
        tauri_build::Attributes::new().app_manifest(tauri_build::AppManifest::new().commands(&[
            "startup_snapshot",
            "retry_startup",
            "choose_models_directory",
            "reset_models_directory",
            "read_ui_storage",
            "set_ui_storage_value",
            "replace_ui_storage",
        ]));
    tauri_build::try_build(attributes).expect("failed to run tauri-build");
}
