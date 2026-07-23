# Workspace CLI review evidence

Captured on Windows from feature commit `329c536d9203cd6cd647197b3c86f7fa78ef7669`.
The PNGs contain rendered CLI receipts; they are not product UI mockups. The lifecycle output is
verbatim. The conversation image condenses the long model prose while preserving the observed file,
settings, thread ID, tool call, commands, and durable transcript result.
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

## Real-model CLI workflow

The feature binary loaded the exact `Qwen3-4B-Q4_K_M.gguf` model on a fresh loopback server. All 36
layers were CUDA-resident. A disposable folder contained `README.md`, `auth.toml`, and `.env.example`.

The following CLI-only workflow was exercised:

```powershell
camelid workspace ask . "Which files configure authentication?"
camelid workspace ask . "What changed?" --thread workspace-0f2824ce-7285-4cee-94e6-b3511c96b7e3
camelid workspace threads .
camelid workspace show workspace-0f2824ce-7285-4cee-94e6-b3511c96b7e3 --workspace .
camelid workspace compact workspace-0f2824ce-7285-4cee-94e6-b3511c96b7e3 --workspace .
camelid workspace compact workspace-0f2824ce-7285-4cee-94e6-b3511c96b7e3 --workspace . --undo
camelid workspace delete workspace-0f2824ce-7285-4cee-94e6-b3511c96b7e3 --workspace .
```

The first ask called `read_file(auth.toml)` and persisted the observed values
`provider="local"`, `session_ttl_minutes=60`, and `require_mfa=true`. The follow-up reused the same
thread ID; `threads` reported two turns; `show` returned both durable turns. Manual compaction archived
two turns, undo restored the previous state, and delete succeeded after a second disposable thread
became active. The server intentionally refuses deletion of the currently active thread.

See `cli-conversation.png` and `cli-lifecycle.png`.

## Checksums

```text
3c5663363c362ca92b7304a5d536bcdb343fef7cbd88bfcad1e664355afebeef  cli-surface.png
eedf80844a044fea6c53f86b9ff23fccca03459e89e54671abd55018b5f6057d  auth-boundary.png
2ff1082b10da53b3387165e3bce6a357c9dc8803e0519eb2b7368a287b44cdc2  cli-conversation.png
8bb5699e3de48ca28ce7f94ca962945af29ae93cabb26ae9b1f8b5c3aee53167  cli-lifecycle.png
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
