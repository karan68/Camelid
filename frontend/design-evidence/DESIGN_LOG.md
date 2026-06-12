# Camelid Frontend Overhaul — Design Log

One entry per phase: what changed, what was tried and rejected, and any
deviation from the build spec with reasoning.

---

## Phase 1 — Design system and visual identity (2026-06-12)

Branch: `feat/frontend-phase-1-design-system` (on top of Phase 0 baseline `c837d05`).

### Concept

Instrument panel, not chat toy. The identity is carried by three things:
1. **The Evidence Chip** (`components/ui/EvidenceChip.jsx` + `lib/evidenceStatus.js` +
   `styles/evidence.css`) — every support/evidence claim now renders through one
   component with a row-scoped mono label, a state icon (color-independent), and a
   click-to-verify popover citing the claim source (capability row id, scope, contract
   copy). Presentation-only: it displays gate state, never computes it.
2. **The status color doctrine** — copper is reserved exclusively for
   supported/verified; desaturated amber for evidence-only/bounded; cool steel blue
   for interactive/informational; muted slate for unsupported (a normal state, never
   alarming); red for errors only.
3. **Mono as a first-class citizen** — IBM Plex Mono carries row ids, statuses,
   endpoint paths, pin-badge evidence lattices, and chip labels.

### Token table (dark canonical / light override)

| Token | Dark | Light | Role |
| --- | --- | --- | --- |
| `--color-bg` | `#0e1216` | `#f6f8fa` | near-black blue-grey base |
| `--color-bg-elevated` | `#141a21` | `#ffffff` | cards, popovers |
| `--color-surface-strong` | `#1c242e` | `#e1e8ef` | strongest panel |
| `--color-text` | `#dde5ed` | `#1b2530` | body text |
| `--color-text-muted` | `#9caab9` | `#4d5d6d` | secondary |
| `--color-text-faint` | `#8190a0` | `#56656f` | captions (AA-fixed) |
| `--color-accent` | `#8fb6dc` | `#2b5c84` | steel blue, interactive/info |
| `--color-verified` | `#dfa371` | `#96531c` | **copper — supported/verified only** |
| `--color-evidence` | `#cfb56a` | `#75601a` | desaturated amber — bounded evidence |
| `--color-planned` | `#8b9aab` | `#546576` | planned/groundwork/target |
| `--color-unsupported` | `#8694a3` | `#5a6a79` | calm honest unsupported |
| `--color-ready` | `#8cc9a0` | `#20713c` | operational runtime health |
| `--color-warning` | `#d8b86a` | `#82610e` | operational warnings |
| `--color-error` | `#e9928a` | `#b3261e` | errors only |
| `--font-display` | Space Grotesk Variable | — | view titles, wordmark |
| `--font-ui` | Inter Variable | — | body |
| `--font-mono` | IBM Plex Mono 400/500/600 | — | ids, statuses, paths |
| `--radius-sm/md/lg/xl/2xl` | 6/8/12/16/20px | — | tightened from 8/12/18/24/32 |

Each evidence color also has `-soft` (fill) and `-border` variants. Full list in
`src/styles/tokens.css`. `scripts/contrast-check.mjs` (`npm run smoke:contrast`)
asserts WCAG AA for all 124 text/status-on-surface pairs in both themes — passing.

### What changed

- `tokens.css` rewritten dark-first: dark is the canonical `:root` palette;
  light is the `[data-theme="light"]` + `prefers-color-scheme: light` override
  (the two light blocks stay byte-identical, mirroring the old dark-block rule).
  All legacy variable names kept so every existing sheet still renders.
- Theme default changed from `system` to `dark` (`useTheme.js`); cycle order is
  now dark → light → system. System-following still works exactly as before.
- Fonts self-hosted via Fontsource (`main.jsx` imports); the Google Fonts CDN
  `@import` was removed. **Baseline correction:** BASELINE.md §7 claimed the
  baseline had "no CDN calls" — wrong; `tokens.css:11` imported Plus Jakarta
  Sans/Outfit from fonts.googleapis.com at runtime. Phase 1 makes the offline
  claim actually true.
- Evidence Chip replaced ad-hoc claim renders in: TopBar (support gate, now
  visible on every tab, not just chat), chat composer status strip, ApiView
  (feature rows, compatibility rows, selected-model contract), SystemView
  (rows + guarded features), AnalyticsView (guarded rows/features), ModelsView
  (tracked-row status pills, 3B acceptance card, external-routing pill).
  Operational model-state pills (downloading/loaded/needs-attention) stay
  `status-pill` — restyled mono/uppercase but still operational-green: runtime
  state is not a support claim and must not look like one (I4 in reverse).
- `pin-badge` evidence lattices restyled to mono micro-labels; their "ready"
  tone now maps to evidence amber, not green — a passing bounded pack is
  bounded evidence, not blanket readiness.
- Brand sparkle re-colored from the legacy purple Gemini-style gradient to the
  instrument gradient (steel → brass → copper), matching `--camelid-aurora`.
- Shell: hairline-bordered glass top bar (56px) with the gate chip + model
  button; display-face titles; tightened radii; backend-unreachable
  empty/error states added to ApiView and SystemView (the views that had none).
- New tooling: `scripts/contrast-check.mjs`, `scripts/capture-views.mjs`
  (puppeteer-core against system Chrome; seeds `camelid-theme` before boot;
  forces a fresh document load per view because hash routing is mount-time).
  `puppeteer-core` added as devDependency; fonts are the only new runtime deps.

### Tried and rejected

- One warm accent for both "supported" and "evidence": rejected — the spec's
  distinction (copper vs desaturated amber) is the product's honesty made
  visible; merging them re-blurs supported vs bounded-evidence.
- Keeping light as the canonical `:root` palette: rejected; dark-first means the
  canonical definition is dark, and it also makes the no-attribute default dark.
- Uppercase mono for all pin-badges: rejected as too loud across 100+ badges;
  pin-badges stay lowercase mono, only EvidenceChip/status-pill labels are caps.
- Replacing every pin-badge with EvidenceChips: deferred — the per-lane evidence
  lattice in ModelsView is Phase 4's drill-down material; converting 100+
  badges now would duplicate that work without the contract-driven popover data.

### Gate results

- `npm run build` clean — JS 145.85 kB gz (baseline 143.76; Phase 7 budget 229.9),
  CSS 19.45 kB gz. Fonts ship as separate self-hosted woff2 assets.
- Smokes: streaming, model-state, capability-readiness, 3b-closure, integration,
  observatory all PASS; `smoke:tiny` PASS against a live backend with the
  unsupported tiny fixture — chat verifiably stays blocked, with the topbar and
  composer chips honestly reading "no matching COMPATIBILITY.md row".
- Readiness-gate logic: `git diff` over `chatGate.js`, `capabilities.js`,
  `modelState.js`, `capabilityReadiness.js` is empty — byte-identical to the
  Phase 0 record.
- `smoke:ui`: unchanged pre-existing failure (stale README-copy assertion).
  Deeper finding while reading it: everything after that first failing assertion
  has been dead for a while — the tail reads `src/styles/components.css` (deleted
  before this overhaul) and asserts pre-redesign TopBar internals
  (`exactHintDetail`) that no longer exist. The smoke needs a deliberate
  re-baseline as its own change; nothing was deleted or weakened in Phase 1.
- Contrast: all 124 pairs AA in both themes (`npm run smoke:contrast`).
- Screenshots: `design-evidence/phase-1/` — all 10 views × dark/light × 1440/390.
  Self-critique fixes during capture: purple sparkle (fixed), capture script
  initially re-shot the chat view 40× because hash navigation doesn't remount
  (fixed in the harness). Known carry-over: observatory run-details panel still
  overflows at 390px (recorded at baseline; Phase 7 responsive audit scope).

---

## Phase 2 — Chat experience (2026-06-12)

Branch: `feat/frontend-phase-2-chat-experience` (pre-work commits 547a233 + 755a7b1,
feature commit follows this entry).

### Pre-work (committed separately)

- BASELINE.md errata appended (offline-fonts claim, smoke:ui health overstatement).
- Deleted three zero-importer pre-redesign orphans (AppSidebar, ConversationDeleteDialog,
  GlobalNotice); scrubbed stale components.css comment references.
- capture-views.mjs gained a sha256 self-check that fails the run when two captured
  views are pixel-identical (the 40-identical-screenshots failure mode); negative-tested.
- smoke:ui re-baselined: full port/retire ledger in commit 755a7b1; negative-tested
  (injected copper-token violation fails the run). It is now in the standing gate set.

### What shipped

- **Markdown**: tables (header detection + instrument-grid styling), ordered lists with
  preserved start numbers, links (http/https/mailto only — any other scheme degrades to
  visible plain text), italics/strikethrough, and per-language syntax highlighting
  (python/rust/bash/json families joined js/html/css) — all still rendered as React
  elements, so there is no innerHTML path at all. New `smoke:markdown` (SSR via vite
  ssrLoadModule) covers tables/lists/links/highlighting/injection-escaping.
- **Metadata footer** on completed assistant messages: model id, the Evidence Chip for
  the row that was active at send time (row id + status snapshot, never paths), token
  counts labeled `usage` (backend) vs `usage est.` (client estimate), TTFT, tok/s,
  duration, and a persistent CLIENT-MEASURED tag (I4).
- **Message actions**: copy (existing), regenerate (truncates the thread at the prior
  user turn and resends through the same gate-checked sendMessage path — no second send
  path exists), edit-and-resend on user rows (inline textarea, Enter resends, Esc cancels).
- **Conversation export** (Markdown/JSON) from Chat history: field-whitelist serializer
  (`lib/conversationExport.js`) so filesystem paths are excluded by construction; smoke:ui
  now feeds it a conversation salted with path fields and asserts none survive (I7), plus
  the telemetry-not-evidence note in both formats.
- **Generation controls drawer**: system-prompt editor with local presets (leads the
  request; the code-first policy prompt appends behind it), and the sampling lane —
  every parameter renders as a guarded "no contract row" Evidence Chip because
  /api/capabilities advertises no sampling rows (BACKEND_ASKS.md #1). The unlock path is
  fully wired (`lib/samplingContract.js`: exact-id row match, per-model persistence,
  contract-gated request overrides) but inert until the contract grows.
- **Keyboard**: Enter/Shift+Enter (existing), Esc cancels stream (existing), Cmd/Ctrl+K
  stub jumps to the composer and says the palette ships in Phase 7 (no fake palette).

### Tried and rejected

- marked/DOMPurify/highlight.js: rejected — the renderer is already React-element-based
  (sanitized by construction); extending it keeps runtime deps at zero and the offline
  property trivial.
- Editable sampling controls with a "values are experimental" disclaimer: rejected —
  I3 says guarded surfaces, not caveated live ones. Controls unlock per-parameter only
  when the contract advertises the exact row.
- A second "regenerate" request path in the hook: rejected — regenerate/edit-resend
  reuse sendMessage with truncate+override options so the chat gate, code-first policy,
  streaming, and abort handling stay single-sourced.

### Gate results

- Build clean; JS **150.66 kB gz** (Phase 1: 145.85; ceiling 229.9), CSS 20.03 kB gz.
- All 10 smokes green: streaming, model-state, capability-readiness, 3b-closure,
  integration (one regex made markup-tolerant for the new keyword highlighting — the
  escaped-content assertion is intact), observatory, **markdown (new)**, **ui
  (re-baselined, now standing)**, contrast, tiny (chat verifiably blocked for the
  unsupported fixture).
- Readiness-gate libs: empty git diff.
- Live manual pass against the loaded supported 3B row, driven through the real UI
  (p2-manual-*.png): chat unlocked; mid-stream abort renders the interrupted warning
  and keeps partial content; metadata footer renders with TTFT 336ms / tok/s / supported
  chip; regenerate replaced the reply without duplicating turns. Structured SSE
  `event: error` mid-stream stays covered by smoke:streaming + smoke:integration
  (the backend offers no way to trigger one on demand — noted, not hand-waved).
- Screenshots: chat + history × dark/light × 1440/390 via the harness (self-check
  passed, 8 distinct) + live-stream evidence set + controls-drawer shot.

---

## Phase 3 — Model management (2026-06-12)

Branch: `feat/frontend-phase-3-model-management`.

### What shipped

- **Card-level Evidence Chips on local GGUFs** (`ModelCardEvidence` in ModelsView):
  every local model card — both "Local runtime" and "Still needs setup" — resolves its
  exact model/quant against the live contract. Matched rows show their real status
  chip; unmatched models get the calm muted "no exact supported row" chip plus a
  "view the compatibility ledger" jump (currently #api; re-targets to the Phase 4 view
  when it exists). Not an error state (I2).
- **Model inspector drawer** (`components/models/ModelInspector.jsx`): fetches
  `/api/models/current` + `/api/models/tokenizer` on open. File section (path with a
  "local-only display; never exported" note, GGUF version, quant from file_type,
  tensor count/offsets, model-native context length explicitly caveated against the
  bounded-pack contract), tokenizer section (model, vocab size, special ids, config
  flags), and the full 35-key KV grid with long values summarized client-side (the raw
  payload is 5.6 MB of vocab/merges arrays — rendered as "[…, N items]"). A banner
  chip pins the framing: descriptive metadata — not support evidence (I2/I4).
- **Tokenizer playground** (`components/models/TokenizerPlayground.jsx`): live
  encode/decode against `/api/models/tokenizer/{encode,decode}` (feature row
  `tokenizer_encode_decode`, cited by the panel chip). Text → token count, per-token
  id+piece chips (per-id decode — faithful for BPE since decode is a fixed id→bytes
  map; capped at 200 with an honest truncation note), add_special/parse_special
  toggles, and a byte-exact round-trip verdict computed over the full sequence.
  Works whenever a tokenizer is loaded — chat support not required, and the chip copy
  says token output does not widen generation support.
- **Load/switch flow**: already had typed-guardrail error surfacing
  (getGuardrailErrorMessage → load_error + notice) and busy states; unchanged. The
  active model's "unmistakable everywhere" treatment comes from the Phase 1 topbar
  gate + composer chips + the active-model-card highlight.

### Tried and rejected

- Storing `/api/models/current` in dashboard state for the inspector: rejected — the
  payload is 5.6 MB; the drawer fetches on open and summarizes immediately instead of
  keeping vocab arrays resident in React state.
- Per-token pieces via incremental prefix decodes: rejected — O(n) requests with no
  correctness gain over per-id decode for BPE; per-id chunks of 16 keep it simple and
  the full-sequence round-trip still catches any normalization drift.
- Evidence chips on catalog-preview cards: deferred — catalog entries are not local
  GGUFs; their "Catalog quant:" labels already stay non-promotional, and Phase 4's
  ledger view is the right home for browsing claims.

### Gate results

- Build clean; JS **153.38 kB gz** (Phase 2: 150.66; ceiling 229.9).
- All 9 offline smokes green + `smoke:tiny` PASS; smoke:ui extended with Phase 3
  assertions (card chip presence + calm no-row copy + ledger link; inspector labeled
  not-support-evidence and barred from gate computation; playground cites
  tokenizer_encode_decode and disclaims generation support; both new components in
  the brand-hygiene sweep).
- Readiness-gate libs: empty git diff.
- Wrong-row demonstration through the real UI: tiny fixture loaded
  (generation_ready=true) → composer reads "Runtime ready, support gated ·
  tiny-generation · No matching COMPATIBILITY.md row", send stays locked with a
  draft present, and the library card shows the calm unsupported chip + ledger link
  (chat-blocked-tiny / library-blocked-tiny screenshots).
- Live checks against the loaded 3B: inspector renders 35 KV rows with arrays
  summarized and context length present; playground round-trips
  "Hello Camelid, parity is the product." at 10 tokens, byte-exact ✓.
- Screenshots: library × dark/light × 1440/390 (harness self-check passed) +
  inspector + playground + blocked-state evidence.

---

## Phase 4 — Compatibility & evidence explorer (2026-06-12)

Branch: `feat/frontend-phase-4-compatibility-explorer`. The signature view.

### What shipped

- **views/CompatibilityView.jsx** — the live ledger, new first-class tab
  (`#compatibility`, registered in HASH_TABS/VALID_TABS/TopBar/sidebar). Everything on
  the screen is the `/api/capabilities` payload at render time: the support-contract
  block (current gate / support policy / unsupported policy verbatim), a stat strip
  (14 rows · 9 supported · 5 "tracked, honestly not claimed"), and one ledger row per
  exact lane. Smoke-enforced: the view source contains zero hardcoded row ids or
  support statuses, and the integration smoke renders it against a mock contract
  (rows come from the mock; trap fields like broad_family lists must NOT render) and
  against a null contract (fail-closed "Ledger unavailable", zero rows).
- **Proven / Not claimed at equal visual weight** — two same-width columns per row;
  the not-claimed column renders the row's `full_support_blockers` copy verbatim with
  `support_scope` underneath. Supported rows get a copper left edge (rule lives in
  evidence.css — the copper-reservation smoke caught it in views.css, which is exactly
  what that assertion is for).
- **Per-row drill-down** — a 13-track evidence checklist (metadata/tokenizer/tensors/
  generation/prompt-token parity/frontend load/template-shape pack/bounded 512–8192
  context packs/perf-RSS), each an Evidence Chip citing the row id plus the
  `*_pack_id` evidence-bundle identifier where the contract advertises one; latest
  checked bucket → result, and the row's readiness-gate sentence.
- **Promotion path** — for non-supported rows only, the contract's `next_step` copy in
  a dashed planned-tinted panel, captioned "an honest checklist, not a promise."
- **Cross-linking** — every Evidence Chip in the app now carries "View in the evidence
  ledger →" in its popover whenever it cites a row id, dispatching a
  `camelid:open-ledger` event the app shell listens for (no prop drilling through
  dozens of chip sites). The ledger scrolls to, highlights, and auto-expands the row;
  api-feature ids resolve to the ledger's feature section. ModelCardEvidence's
  "view the compatibility ledger" link re-targeted from #api to the new view.
- **"How to read this ledger" explainer** in product voice: exact-row support, bounded
  packs ≠ native context, perf ≠ throughput promises, unsupported is a normal state.

### Tried and rejected

- Hash-fragment row addressing (#compatibility/<row>): rejected — hash routing is
  mount-time-only here; the event + focus-state approach deep-links from live chips
  without rearchitecting navigation.
- Rendering manifest paths from README/COMPATIBILITY.md copy: rejected — that would be
  doc-derived support claims the contract doesn't make. The contract exposes pack ids
  (cited); manifest references are BACKEND_ASKS.md #2 and render automatically when
  `*_pack_manifest` fields appear.

### Gate results

- Build clean; JS **156.08 kB gz** (Phase 3: 153.38; ceiling 229.9).
- 10/10 smokes green (integration + ui extended with the ledger assertions above);
  readiness-gate libs empty diff; `smoke:tiny` still proves fail-closed chat.
- Live UI checks: 14 rows render from the live contract with not-claimed on every
  row; drill-down shows 13 tracks with real pack ids; deep-link from a library
  tracked-row chip lands focused + auto-expanded on llama32_3b_instruct_q8_0. One
  honest negative: the composer chip with the tiny fixture loaded cites no row, so it
  correctly offers no ledger link.
- Screenshots: compatibility × dark/light × 1440/390 (self-check passed) + drill-down
  + deep-link focus shots in design-evidence/phase-4/.

---

## Phase 5 — API workbench (2026-06-12)

Branch: `feat/frontend-phase-5-api-workbench`.

### What shipped

- **components/api/ApiWorkbench.jsx + lib/apiExamples.js** — nine routes of the live
  surface as workbench cards: /v1/health, /v1/models, /v1/chat/completions,
  /v1/completions, /api/capabilities, /api/models/tokenizer/encode, and the three
  fail-closed routes (/v1/embeddings, /v1/responses, /v1/messages) rendered as typed
  guarded rows citing the fail_closed_native_compatibility_routes feature row.
- **Examples** in curl / Python / JS-fetch, pre-filled with the live API base and
  loaded model id, copy button per card. Generation examples mirror the chat request
  shape (greedy temperature=0, streaming). Gate evidence: the chat-completions curl
  example was extracted from the rendered page DOM and executed verbatim — it streamed
  SSE chunks from the live 3B.
- **Try-it gating (I1/I3)**: generation endpoints run only when the shared exact-row
  chat gate is green — including /v1/completions, where a found sharp edge made this
  matter: the raw backend route answers for ANY loaded model (it generated <unk> tokens
  from the unsupported tiny fixture), so the workbench card states explicitly that the
  route answers but the UI keeps generation examples gated like chat. Read-only routes
  run whenever the backend answers; fail-closed routes never run. Verified live in both
  directions: tiny fixture → both generation try-its guarded with typed copy while
  health stayed runnable; 3B loaded → chat try-it unlocked.
- **Request inspector (I4)**: rendered request, status, headers/total timings, pretty
  JSON bodies, and a timestamped SSE chunk log (capped at 80 lines with an honest
  truncation note) under a pinned "operational telemetry — not compatibility evidence"
  chip. Live run: 9 chunk lines, headers 162ms, total 1100ms.
- ApiView's four static endpoint cards were absorbed into the workbench; the
  readiness-gated curl block and every smoke-asserted gate string stayed live (one
  asserted phrase was restored into the section copy when the cards went away — the
  smoke caught it).

### Tried and rejected / deviations

- "Python · openai sdk" as the visible tab label: rejected by the pre-existing
  integration-smoke brand assertion on rendered markup. Deliberate resolution: the tab
  label is just "Python"; the SDK example itself (which must name the class it
  instantiates — that is technical compatibility content, not product copy) renders
  only when the tab is selected, so the default markup stays brand-clean.
  lib/apiExamples.js is consciously excluded from the brand sweep with an inline
  comment; the workbench component itself remains swept.
- Gating /v1/completions as 'tokenizer-level' (it technically runs for any loaded
  model): rejected — I1 says generation examples gate exactly like chat; a route that
  emits unsupported-model tokens is precisely what the gate exists to keep out of the
  paved path.

### Gate results

- Build clean; JS **159.41 kB gz** (Phase 4: 156.08; ceiling 229.9).
- 10/10 smokes green; integration smoke extended with both gating directions
  (blocked fixture → data-tryit-ready=false for both generation routes + typed copy +
  health true; green 3B fixture → chat try-it true); smoke:ui extended with workbench
  assertions. Readiness-gate libs: empty diff.
- Copy-paste gate: page-extracted curl ran verbatim against the live backend (SSE).
- Screenshots: api × dark/light × 1440/390 (self-check passed) + gated-state +
  SSE-inspector shots in design-evidence/phase-5/. Backend left re-gated on the tiny
  fixture (smoke:tiny re-run green at close).

---

## Phase 6 — Observability dashboard (2026-06-12)

Branch: `feat/frontend-phase-6-observability`.

### What shipped

- **lib/telemetryLog.js** — the session store. Records arrive ONLY from real traffic:
  the chat send path (success, interruption, and error all record), workbench try-it
  runs, and the live health polls. In-memory ring buffers (500 requests / 240 polls),
  nothing persists across reloads, no seeding path exists (smoke bars the obvious
  fabrication routes and the empty state promises "never seeds or invents data").
- **views/TelemetryView.jsx** (`#telemetry`, first-class tab): summary tiles
  (requests, error rate, median TTFT / tok/s / duration — all client-measured and
  labeled), SVG sparkline trends, per-model breakdown by model id (captioned: grouping
  implies nothing about support), backend reachability strip, and the request log.
- **Request log**: time, endpoint, model, outcome, duration, token counts; prompt
  content REDACTED by default with a per-session reveal toggle; Export JSON goes
  through a field whitelist that cannot include prompt content or paths — smoke-
  enforced behaviorally with a salted record (secret prompt + /Volumes path → absent
  from export, whitelisted fields + not-evidence note present).
- **Health with backoff**: the dashboard refresh loop became self-scheduling — 2.5s
  while the backend answers, doubling to a 20s ceiling on consecutive failures, reset
  on success. Every poll outcome (latency or failure) lands in the reachability strip.
- **I4 pinned everywhere**: page-level chip + per-panel captions; a smoke assertion
  bars perf numbers from rendering inside Evidence Chips in this view. Bounded
  perf/RSS contract evidence stays in the Compatibility ledger, explicitly pointed to.

### Tried and rejected

- Persisting telemetry to localStorage: rejected — "session metrics" should die with
  the session; persistence would also turn yesterday's numbers into ambient pseudo-
  evidence.
- Folding into AnalyticsView: rejected — Analytics is conversation usage over stored
  history; this is live operational traffic. Mixing them blurs the I4 boundary the
  spec draws.
- A separate health poller for the panel: rejected — the dashboard already polls; a
  second poller would double traffic and make the history lie about cadence. The
  existing loop gained backoff instead.

### Gate results

- Build clean; JS **163.18 kB gz** (Phase 5: 159.41; ceiling 229.9).
- 10/10 smokes green incl. the behavioral export-whitelist check; gate libs empty diff.
- "Demonstrably real requests" shown live end-to-end in one browser session: fresh
  session renders the empty state with only real health polls in the strip; one real
  chat send (3B, "telemetry check") + one workbench health try-it populate exactly 2
  log rows, the tiles (TTFT median 284ms · 3.9 tok/s · 387ms), and a 3B per-model row;
  prompt redacted by default, reveal toggle shows it. Failure history demonstrated in
  an isolated profile against a dead API base: 4 failed polls in 16s (backoff visibly
  stretching the cadence), red strip cells.
- Screenshots: telemetry × dark/light × 1440/390 + populated/empty/unreachable shots
  in design-evidence/phase-6/.

---

## Phase 7 — Polish, command palette, accessibility, performance (2026-06-12)

Branch: `feat/frontend-phase-7-polish`. The closing phase.

### What shipped

- **Command palette** (Cmd/Ctrl+K): navigate all 12 views, new conversation, theme
  cycle, switch model (hint stays gate-honest: "readiness still gates send"), and jump
  to any compatibility row — reusing the same `camelid:open-ledger` event the chips
  use, live-verified to land focused on the row. Combobox/listbox semantics, arrow/
  enter/esc keyboard model.
- **"?" shortcut overlay** documenting the full keyboard map (outside text fields).
- **Accessibility**: Lighthouse a11y **100 on chat, 98 on compatibility** (gate ≥95).
  The one real fix it surfaced: interactive Evidence Chips now carry an explicit
  aria-label (the topbar gate chip loses its visible label at mobile widths and was
  name-less). Composer status strip became a polite live region; heading order
  normalized in the ledger; icon-per-state chips (Phase 1) already satisfied
  color-independence.
- **Performance**: route-level code splitting — chat stays eager, the other 11 views
  load on first visit. Initial JS chunk **104.10 kB gz** (was 163.32 monolithic);
  total across all chunks **176.39 kB gz** vs the 229.9 budget (1.6× the 143.76
  baseline — met with 23% headroom). Long-conversation windowing (latest 60 turns +
  "show earlier" expander; the telemetry log was already windowed) instead of a
  virtualization dependency. Fixed an ineffective dynamic import in the poll loop.
- **Responsive**: the baseline-recorded observatory run-details overflow at 390px is
  fixed (panel stacks under the canvas ≤700px; live-measured 0px horizontal overflow).
  Full audit captured at 390/768/1024/1440.
- **Identity**: SVG favicon (instrument sparkle, steel→brass→copper on the dark base)
  + theme-color meta; wordmark already carried by Space Grotesk since Phase 1.
- **frontend/README.md**: new views/features/shortcuts section with the explicit
  statement that readiness-gate semantics are unchanged (smoke-asserted).

### Tried and rejected

- A virtualization library for message lists: rejected — windowing achieves the
  perf goal with zero dependencies and no scroll-anchoring edge cases during
  streaming.
- Chasing compatibility from 98 to 100: the residual flag is a heading-order
  nit inside contract-rendered sections; restructuring real content hierarchy for a
  scanner point wasn't worth bending the ledger's semantics. 98 ≥ 95 gate.

### Gate results

- 10/10 smokes green (smoke:ui extended with palette/overlay/code-split/README
  assertions); readiness-gate libs: empty diff — byte-identical through all 8 phases.
- Lighthouse a11y: chat 100, compatibility 98 (both ≥95).
- Bundle budget met: 176.39 kB gz total / 104.10 initial vs 229.9 ceiling.
- Final screenshot set: 12 views × dark/light × 1440/390 (48 shots, self-check
  distinct) + 24-shot responsive audit at 768/1024 + palette/shortcuts/observatory-
  fix evidence. design-evidence/phase-7/.

### Before / after (the whole overhaul)

Phase 0 baseline → Phase 7: a Gemini-styled chat shell with ad-hoc status badges and
a Google-Fonts CDN dependency became an instrument-panel operator console with one
claim component (the Evidence Chip, cited everywhere, deep-linked to a live-contract
evidence ledger), a chat surface with telemetry-honest footers and contract-gated
controls, a model inspector + tokenizer playground, a gated API workbench whose curl
examples run verbatim, a real-traffic-only session telemetry dashboard, a command
palette, AA-contrast-smoked dual themes on self-hosted fonts, and a 10-smoke gate
suite (from 8, one of which was dead) — at 104 kB gz initial JS against a 143.76 kB
baseline monolith, with the fail-closed chat gate byte-identical throughout.

---

## Final acceptance (2026-06-12, after Phase 7)

1. `npm run build` clean — initial JS 104.10 kB gz, total 176.39 kB gz. ✓
2. `smoke:streaming`, `smoke:contrast`, re-baselined `smoke:ui` green (with the full
   10-smoke suite). ✓
3. Backend up → `smoke:tiny` green: the unsupported fixture loads, reports
   generation_ready=true, and chat verifiably stays blocked. ✓
4. Supported-row manual pass (Llama 3.2 3B Instruct Q8_0 — the supported
   `supported_exact_row_smoke` row; the spec names TinyLlama, any supported exact row
   satisfies the gate-green condition): load ✓, inspector ✓, streaming chat ✓, Esc
   abort mid-stream renders interrupted state ✓, regenerate completes with telemetry
   footer ✓, conversation export (path-free, smoke-enforced) ✓, workbench try-it
   unlocked + request inspector ✓, telemetry dashboard populating from the session ✓,
   Compatibility ledger matching /api/capabilities exactly (14/14 row ids, 11/11
   feature rows) ✓.
5. frontend/README.md source-of-truth section re-read against the shipped app: every
   listed behavior holds — health/models/capabilities consumption, meta-as-descriptive,
   no native-route unlocks, load via /api/models/load, gate visible in the top bar on
   every tab with a direct jump to the contract (via the ledger), API tab first-class,
   readiness-gated examples, file_type quant normalization, exact-row wins shown
   row-scoped, streaming + typed SSE error handling, and the fail-closed chat gate.
   No drift found. ✓

The overhaul is complete: Phases 0–7 shipped, all invariants I1–I7 held at every
gate, and the readiness-gate libraries are byte-identical to the Phase 0 record.
