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
