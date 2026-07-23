# Workspace CLI review evidence

Captured on Windows from feature commit `329c536d9203cd6cd647197b3c86f7fa78ef7669`.
The PNGs contain rendered transcripts of the exact outputs below; they are not product UI mockups.
No bearer value is shown or stored in this bundle.

## CLI surface

```powershell
C:\camelid-workspace-cli-pr\target\debug\camelid.exe workspace --help
```

See `cli-surface.png`.

## Live authentication boundary

A disposable server was started on loopback port 8295 without a loaded model:

```powershell
camelid.exe serve --addr 127.0.0.1:8295 --no-open
```

The automatically authenticated CLI request crossed Workspace authorization and reached the
model-readiness gate:

```text
camelid.exe workspace --addr 127.0.0.1:8295 threads C:\camelid-workspace-cli-pr
Error: model_not_loaded: load a tool-capable model before starting Workspace
exit=1
```

The same route without the CLI credential was rejected before model or path processing:

```text
GET /api/agent/workspace/threads?workspace=...
HTTP 403
```

See `auth-boundary.png`. The forced-stop smoke credential was removed after capture.

## Checksums

```text
3c5663363c362ca92b7304a5d536bcdb343fef7cbd88bfcad1e664355afebeef  cli-surface.png
eedf80844a044fea6c53f86b9ff23fccca03459e89e54671abd55018b5f6057d  auth-boundary.png
```

## Validation context

The implementation commit was validated before evidence capture with:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- 43 focused Workspace/authentication/transport tests
- 18 Workspace authorization/session tests
- 3 CLI parser tests
- clean-target full default-feature suite: 1,145 library tests passed, 61 ignored; 22 binary tests passed; 89 API tests passed; remaining integration targets passed
- live loopback socket smoke described above

The optional all-feature runtime test attempt compiled successfully but 19 unrelated CUDA tests could
not run because NVRTC was unavailable in the isolated clone. Strict all-feature Clippy passed.
