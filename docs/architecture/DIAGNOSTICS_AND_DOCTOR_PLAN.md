# Runtime diagnostics and `camelid doctor`

Status: proposal (not started). Written 2026-07-28 against `main` @ `f8649c629`.

This document is deliberately product-first. The engineering is easy; the question worth
answering is whether this earns its place, and which third of it actually does.

---

## 1. Verdict up front

**Worth building — but not as one feature, and not in the order it is usually pitched.**

The idea is normally sold as "a support tool: `doctor` emits a bundle you can ask users
for." That framing is the *weakest* part of it. Camelid has 145 stars, 19 forks, 13
issues ever filed, and **zero open issues**. Support load is not a bottleneck today, and
building support infrastructure ahead of support load is a classic way to feel productive
while moving nothing.

What *is* real, right now, at the current user count:

1. **The engine can panic and leave no trace anywhere.** That is a defect, not a missing
   convenience — and it has already happened to a real external user.
2. **A user cannot see which execution lane actually ran.** For a project whose entire
   thesis is *"local AI you can verify,"* that is a hole in the thesis, not in the
   tooling.

Those two are worth doing at zero users. The redacted support bundle is worth doing later,
if ever. Section 6 ranks the slices accordingly; section 9 gives explicit kill criteria.

If the honest answer to "what are the next 90 days" is model breadth or performance, then
build slice S1 only (it is small and closes a real defect) and drop the rest. Nobody has
ever chosen an inference engine because it shipped a `doctor` command.

---

## 2. The problem, as measured (not asserted)

Every claim below was verified in the tree at `f8649c629`.

| Claim | Reality |
|---|---|
| "No structured logging" | **Partly false.** `tracing` + `tracing-subscriber` are already dependencies (`Cargo.toml` 47–48) and initialized in `src/main.rs` 1501–1503 via `EnvFilter::from_default_env()`. |
| "No log file" | **True.** The subscriber writes to the console only. There is no file sink anywhere. |
| "No panic report" | **True, and worse.** The only panic hook in the binary is `quiet_cudarc_loader_panics` (`src/main.rs` 227) — it *suppresses* cudarc loader panics. The most likely failure class on a stranger's GPU box is deliberately silenced. |
| "No `/metrics`" | **True in substance.** The route exists (`src/api/mod.rs` 1880) but is `unsupported_llama_server_metrics`, a fail-closed stub. |
| "No config file" | **True.** Configuration is **319 distinct `CAMELID_*` environment knobs** and nothing that can print them back. |
| "Four v0.4.x patches in ~24h" | **True and understated.** `v0.4.0` 10:37 → `v0.4.4` 19:38, all on 2026-07-24. Five tags in about nine hours. |

Two structural consequences follow from the above.

**The shipped app throws engine output away.** The desktop spawns the sidecar with
`.stdout(Stdio::null())` and `.stderr(Stdio::piped())` (`camelid-desktop/src/engine.rs`
248–249). stderr is drained **only on a startup failure**, to render on the splash screen.
On success, `Engine { child, .. }` retains the child with the stderr pipe attached and
never read again (`engine.rs` 111–112, 272–274). So for a packaged user, once the engine is
healthy, everything it says goes nowhere.

**The effective configuration is unobservable.** 319 environment knobs select backends,
kernels, KV dtypes, fit overrides and lane behaviour. Nothing prints the resolved vector.

---

## 3. Evidence from real users

Thirteen issues have ever been filed. The externally-reported ones are the tell:

- **#302** — *"Concurrent /v1 generation requests corrupt shared decode state (garbled
  output, non-deterministic greedy, intermittent slice panic)."* Contains a real panic:
  `range end index 7340160 out of range for slice of length 7340032`.
- **#310** — *"Garbled streaming output under CUDA temperature sampling (temp > 0);
  non-streaming and greedy are clean."*
- **#469** — *"Can't pull qwen3_4b_q4_k_m."*
- **#254** — *"Windows Fails on Build."*

Read #302 closely, because it is the strongest possible argument *and* the strongest
counter-argument at once.

It is an exceptional report — reproduction steps, a correct root-cause hypothesis about
shared `Arc`s under read locks, and the reporter opened a fixing PR. And yet its entire
Environment section is: *"Camelid CUDA build (`selected_backend: cuda_resident_q8_runtime`,
`cuda_resident_active: true`), Windows x86_64, small Qwen3 Q8_0 model."* No driver version.
No CUDA version. No VRAM figure. No exact artifact filename or SHA. No stack trace for the
panic — just a copied message.

That is the **best case**. A sophisticated reporter, hand-assembling a partial environment
snapshot, because the product offers no way to produce a complete one. The median reporter
will send you a screenshot.

The honest counter-argument: #302 and #310 were both diagnosed and fixed *without* any of
this. So diagnostics are not load-bearing for the sophisticated reporter. Their value is
raising the floor for everyone else — and, in the panic case specifically, creating
information that currently **does not exist anywhere at all**.

---

## 4. Who is actually hurt

**The user on their own machine (primary, and usually forgotten).** Camelid is local-first.
The user owns the hardware, the model file, and the process. When it breaks, they are
entitled to know why *on their own machine*, without filing an issue with a stranger. Today
a packaged install that panics mid-generation shows them nothing and writes nothing. This
persona alone justifies slice S1, independent of any support workflow.

**The maintainer.** Five same-day patch tags, all fixing things only visible in the built
artifact. That loop was closed by dogfooding plus the new pre-release artifact contract
(`scripts/check-release-artifact.mjs`, `smoke-release-artifact.mjs`,
`verify-release-checksums.mjs`, merged via PR #510) — not by user reports. Diagnostics do
not shorten that loop much. Be honest about it rather than claiming the win.

**The contributor / reporter.** See section 3. They are currently doing manual archaeology
to produce a worse answer than the binary could produce automatically.

---

## 5. Why this is a product, not plumbing

The repo already has an evidence system — but only for **models**. The capability ledger,
the compatibility rows, the parity receipts, `camelid verify` with its digest-sealed
report: all of it certifies *artifacts*, in the lab, before the user's machine is involved.

There is **zero** evidence about *executions on the user's machine*. Which lane ran. Which
backend was selected. Whether the model went GPU-resident or silently fell back to CPU.
Which exact artifact SHA was loaded. Whether the run took the verified path or the
experimental one.

That is a real gap in the thesis. `filename_is_supported_exact_row` (`src/api/mod.rs`
20087) will quietly classify a model as `ExperimentalImplemented` rather than `Supported`
because of an exact-filename mismatch, and the user has no accessible way to learn that this
is what happened to them.

So the framing that makes this worth building is not *"a support bundle."* It is:

> **A local execution receipt: the runtime half of the evidence story.**

Model evidence says *"this artifact was certified."* Execution evidence says *"this is what
your machine actually did with it."* One is a lab claim; the other is proof of delivery. A
project selling verifiability that ships only the first half is selling half the product.

This is also the answer to *"is it worth anything later?"* — it is the same primitive the
Enterprise line already converged on independently (per-request serving receipts, config
SHA, attribution headers). Building the OSS-side seed of it now means the two lines share a
concept rather than growing two incompatible ones.

---

## 6. Slices, ranked by value (not by build order convenience)

### S1 — Panic and fatal-error capture to disk. **Do this regardless.**

The smallest slice and the only one that closes an existing defect.

- Install a real panic hook that records payload, thread name, location and (behind
  `RUST_BACKTRACE`) a backtrace to a per-user file.
- Keep the cudarc suppression behaviour for *console noise*, but stop letting it mean
  "record nothing." Suppressed-but-recorded, not silenced.
- Record fatal startup errors the same way, so the desktop's splash-screen message has a
  durable counterpart on disk.

Value: a real user has already hit a real panic (#302). Today that panic leaves no trace on
the machine it happened on. Worth doing at zero users because it is a correctness gap.

### S2 — `camelid doctor`: the effective execution report. **The actual product.**

A subcommand that prints, and can serialize, what the runtime resolved:

- resolved `CAMELID_*` configuration vector (only knobs that are set, plus defaults that
  changed behaviour)
- hardware probe (`HardwareProfile::detect()`), CUDA/driver probe (`src/cuda.rs`)
- selected backend / execution lane, and whether it is the verified or experimental one
- model inventory: exact filenames, sizes, SHAs, lane classification per artifact
  (`Supported` / `ExperimentalImplemented` / `Unsupported`) **and the reason**
- fit verdicts (`src/fit.rs`), including exact-vs-approx dims confidence
- capability ledger summary
- loopback bind check

Almost all of this is **composition of probes that already exist**. The new content is the
lane/config attribution and the report type itself. This is the slice that is brand-aligned
rather than merely useful.

### S3 — On-disk rotating log. **Needed to make S1/S2 durable, but it carries the real cost.**

A bounded file sink, off the console path. This is where the actual design work lives:
rotation, size caps, redaction, and the pipe constraint in section 7. Do not start here
just because it sounds like the foundation.

### S4 — The redacted support bundle (zip). **Optional. Probably later. Possibly never.**

Zero open issues. Build this when a real person is actually unhelpable without it, and let
that person's case define the contents. Building it speculatively means guessing at the
contents of a conversation that has not happened.

---

## 7. Hard design constraints

**7.1 The engine must own the file sink — the desktop must not scrape stderr.**
The desktop retains the child with a piped, undrained stderr (`engine.rs` 111–112,
272–274). Today that is harmless only because logging is effectively silent. Turn on real
stderr logging without changing that, and the OS pipe buffer fills and **the engine blocks
on write** — a hang, in the shipped artifact, caused by the diagnostics feature. The file
sink must be written by the engine process itself, to a known per-user directory.

**7.2 Diagnostics must never be able to break the thing they diagnose.**
Fail open, always. A full disk, a read-only directory, or a permissions error must degrade
to "no log" and never to a failed startup or a failed generation. No `unwrap` on any path
in this subsystem.

**7.3 Redaction is the specification, not the polish.**
The report will contain home-directory paths (usernames), model paths, and anything that
leaks into logs. Prompts and generated text must never reach the log or the report — not
truncated, not hashed-but-recoverable, not "only at debug level." A local-first product
that ships a diagnostics tool which exfiltrates prompts has destroyed its own premise. Write
the redaction rules before the writer, and test them with hostile fixtures.

**7.4 No phone-home. Ever.**
No telemetry, no upload, no opt-out analytics. Output is a file on the user's disk that they
choose to share. This is non-negotiable for the brand and should be stated in the README
line that introduces the command.

**7.5 Bounded by construction.**
Size cap and rotation from the first commit, not as a follow-up. An unbounded log on a
long-running local server is a disk-fill bug waiting for the first heavy user.

**7.6 Keep `src/main.rs` involvement to a thin arm.**
That file is past 7,000 lines and is the worst merge-conflict surface in the repo. All logic
belongs in a new module. (There are currently zero open upstream PRs, so overlap is nil
*today* — that will not stay true.)

**7.7 No new dependencies.**
`tracing`, `tracing-subscriber` (with `fmt` + `env-filter`), `serde_json`, `clap` and
`anyhow` are already present. This slice should add none.

---

## 8. What this explicitly is not

- Not a `/metrics` endpoint. That is a different feature dragging in a Prometheus surface
  and a scrape contract. The existing stub stays fail-closed.
- Not a config *file*. Introducing one means designing precedence over 319 environment
  knobs. `doctor` reads configuration; it does not add a way to set it.
- Not crash-reporting-as-a-service. No Sentry, no upload, no aggregation.
- Not a performance profiler. Timing data belongs to the existing bench and GAIT lanes.
- Not a support claim. A `doctor` report is an observation of one machine; it certifies
  nothing and must never be quoted as evidence in the ledger.

---

## 9. Kill criteria — when to stop or not start

Stop, and say so plainly, if any of these hold:

- **S1 slips past a week of work.** It is a panic hook and a bounded file writer. If it is
  sprawling, the scope has been captured by S3/S4.
- **S2 starts requiring new probes rather than composing existing ones.** The value case
  rests on composition. If it needs a new hardware or CUDA introspection layer, the cost
  model is wrong and the slice should be cut back to what already exists.
- **Redaction cannot be made simple.** If keeping prompts out of the output requires
  auditing every log call site in the codebase, ship S1 (panics only — no user text by
  construction) and abandon S3.
- **The user count stays flat and no one reports an unhelpable bug for a full release
  cycle.** Then S4 is confirmed unnecessary; close it rather than letting it linger as
  perpetual almost-work.

---

## 10. What this does not fix

Worth stating so the feature is not oversold:

- **#254 (Windows build failure)** — happens before a binary exists. Unreachable.
- **#279 (misleading README)** — a documentation problem.
- **#310 / #302 class bugs** — diagnostics would have made the reports *better*, not made
  the bugs *findable*. Both were found by users hammering the product, and no amount of
  logging substitutes for that.
- Anything about model breadth, parity, or performance. This slice wins zero new users. It
  is integrity and insurance, not acquisition. Budget it that way.

---

## 11. How success is judged

Deliberately modest and checkable:

- A panic in a packaged install leaves a readable artifact on the user's disk. (Verify by
  deliberately inducing one in a release build — negative control, per repo practice.)
- `camelid doctor` output answers, without asking the user anything: which backend ran,
  which exact artifact loaded and its SHA, whether that artifact is a supported row and why
  or why not, and which non-default `CAMELID_*` knobs were in effect.
- The next externally-reported runtime bug arrives with a complete environment section that
  the reporter did not have to assemble by hand.
- No regression: startup time, and no new failure mode when the log directory is
  unwritable.

---

## 12. Open questions

1. **Log location.** `%LOCALAPPDATA%\camelid\` matches the existing fit-dims cache
   convention on Windows; the cross-platform equivalent needs a decision that does not add
   a `dirs`-style dependency.
2. **Default log level.** Currently `EnvFilter::from_default_env()` with an unset `RUST_LOG`
   is effectively silent. A file sink needs its own default directive, chosen so that the
   file is useful without being noisy — and section 7.1 means this cannot simply be "turn
   stderr up."
3. **Does `doctor` need a running server?** Composing `/api/capabilities` is easiest against
   a live instance, but the most valuable moment to run `doctor` is when the server will not
   start. It likely needs both modes: offline (probes only) and attached (probes + live
   capability report).
4. **Relationship to `camelid verify`.** Both produce sealed reports about artifacts. Decide
   now whether `doctor` embeds a verify summary or merely points at it, so two overlapping
   report formats do not grow in parallel.
