# Authenticated LAN and remote Chat evidence

This bundle contains two physical Android 16 Chrome receipts. Screenshots 01-07 were captured on 2026-08-20 over an ADB USB reverse to the Windows host loopback listener. That flow used the release binary built for feature commit `7f05d5ccc61c` (SHA-256 `f09e94e446a4484816e7e64147be2e807b7b99893208061d38a0f89ac1a503a4`). Screenshot 08 was captured on 2026-08-22 over private Tailscale Serve HTTPS using the PR #684 binary at `93d22a0e2bf1`.

## Sequence

1. `01-terminal-lan-key-redacted.png` shows live `camelid lan-key` CLI output. The exact credential value and user-specific path were redacted before the terminal rendered; the generated start command was wrapped for legibility.
2. `02-mobile-api-key-required.png` shows the physical browser reaching Camelid but receiving the real API-key-required state.
3. `03-mobile-authenticated-chat.png` shows the same browser authenticated and returned to Chat.
4. `04-mobile-chat-only-drawer.png` shows the restricted LAN Chat navigation: Chat, Chat history, Memory, and Settings; administrative/model-management views are absent.
5. `05-mobile-safe-model-selector.png` shows direct model-file choices from the configured models directory and the resident model readiness state.
6. `06-mobile-3b-switch-success.png` shows the physical browser after switching to `Llama-3.2-3B-Instruct-Q4_K_M.gguf`.
7. `07-mobile-3b-chat-verified.png` shows the deterministic prompt, exact `CAMELID_MOBILE_3B_OK` response, 3B Q4 model identity, and client-measured VERIFIED receipt.
8. `08-mobile-remote-cellular-redacted.png` shows the physical phone on cellular data with the Tailscale VPN indicator, completed response receipt metrics, and the real 1B Q8 model still loaded and ready. Wi-Fi was disabled, mobile data was attached, and `adb reverse --list` was empty during this receipt. The private tailnet hostname in Chrome's address bar is the only redacted pixel region; it was replaced with `Private Tailscale HTTPS` after capture.

No server-terminal PNG is included. Its window capture was unreliable and was omitted rather than represented by a staged or ambiguous screenshot. Server readiness is demonstrated by the authenticated physical-browser flow, successful model switch, completed streamed inference, and visible receipt.

The remote receipt completed a real streamed `REMOTE_CELLULAR_OK` response before this frame was retained. The exact response text is above the visible crop, so the screenshot is evidence of the physical cellular/Tailscale path and completed receipt metrics, not standalone evidence of the response string. The accompanying PR validation record carries the command and lifecycle observations: one private proxy, Funnel absent, authenticated Chat, scoped stop, and a still-ready loopback backend.

No API key, account email, device serial, Tailscale address, private hostname, private LAN address, or user-specific filesystem path is stored in this directory. The redacted PNG has no text, EXIF, or compressed-text metadata chunks. `SHA256SUMS` covers the evidence files.
