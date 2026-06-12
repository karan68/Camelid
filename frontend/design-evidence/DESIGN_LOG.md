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
