# Workspace CLI review evidence

Captured on Windows from feature commit `329c536d9203cd6cd647197b3c86f7fa78ef7669`.
The `raw/` directory contains direct command stdout/stderr receipts from the feature binary.
No bearer value is shown or stored in this bundle.

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

See:

- `raw/01-ask.stdout.txt`
- `raw/02-threads-show-compact-undo.txt`
- `raw/03-resume.txt`

Delete was exercised separately after moving active ownership to a disposable second thread, but it
is not claimed by the raw receipt bundle because its standalone capture wrapper became unreliable.

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
