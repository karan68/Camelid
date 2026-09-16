# README screenshots

Captured on 2026-09-16 from the frontend at source commit `45a20f0f` (v0.7.7).
These are actual browser renders with isolated, illustrative demo data, not live
model outputs, benchmark results, or additional model-support evidence.
No personal conversations, credentials, or local project contents are used.

| Images | Capture |
|---|---|
| `desktop-chat.png`, `desktop-files.png`, `desktop-projects.png` | Dark theme, 1440 × 960, full local UI |
| `mobile-chat.png`, `mobile-files.png`, `mobile-context.png` | Dark theme, 390 × 844, touch/mobile viewport, `lan_chat_only` fixture |
| `desktop-code.png` | Dark theme, 1440 × 1050, coding browser smoke's file-approval fixture |

Mobile images emulate a phone viewport in Chromium; they are not physical-device
or Safari validation. History, project context, model state, and coding activity
are fixtures. The chat capture reads the repository's compatibility ledger without
changing its rows. Network requests outside the isolated capture server are blocked.

From `frontend/`, with dependencies and Chrome or Edge installed:

```sh
npm run build
node scripts/capture-readme.mjs
node scripts/coding-browser-smoke.mjs
```

The first capture script writes six PNGs here and checks for page errors and
horizontal document overflow. For the coding image, copy
`target/coding-browser/coding-approval-dark.png` (relative to the repository root)
to `docs/assets/readme/desktop-code.png`. The coding smoke owns its fixture and
checks approval controls, agent activity, saved-session behavior, and responsive
layouts. Review all images visually after capture before updating the README.
