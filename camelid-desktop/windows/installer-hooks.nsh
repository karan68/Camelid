; Camelid Desktop - NSIS installer hooks.
;
; PROBLEM. An in-place NSIS upgrade overwrites the files the new version ships, but it
; never removes a file an OLDER version installed that the current one no longer ships.
; Such a file strands in $INSTDIR forever: re-running the documented one-command
; installer cannot clear it, because the installer only ever writes.
;
; The case that motivated this: releases built before the `.alt.dll` filter landed in
; scripts/package-windows-cuda.ps1 staged `nvrtc64_120_0.alt.dll` into sidecar\ - an
; 85.7 MB alternate JIT backend cudarc never loads. Current releases correctly omit it
; (the filter, plus the `.alt` forbidden rule in scripts/check-release-artifact.mjs), so
; nothing in the install path was ever going to overwrite or remove it. Every user who
; installed before that filter is still carrying the dead 85.7 MB.
;
; FIX. Delete the NVRTC redistributables from sidecar\ before the file copy, so the
; install re-lays exactly the set THIS version ships. Deleting the whole family - rather
; than blacklisting `.alt` - is what makes this self-healing: NVRTC filenames encode the
; CUDA version (nvrtc-builtins64_129.dll is CUDA 12.9; nvrtc64_120_0.dll is the 12.x ABI),
; so a CUDA_VERSION bump renames them and would strand the previous ones exactly the same
; way. A rename is precisely the case an overwrite-only installer cannot fix.
;
; SCOPE - deliberately narrow; do NOT widen this to clear sidecar\ wholesale. sidecar\ is
; not purely shipped content: the desktop's model store is the `models\` folder BESIDE the
; engine binary (`sidecar_models_dir` in ../src/engine.rs), i.e. sidecar\models\, which
; holds multi-GB user-downloaded GGUF weights. Only the nvrtc* DLLs that
; scripts/package-windows-cuda.ps1 stages are shipped content safe to remove here.
; tests/installer_hooks.rs enforces that scope in CI.
;
; FAILURE MODE is benign by construction. `Delete` on a missing file is a no-op, and on a
; locked file (the sidecar is running and has the DLL mapped) it sets the error flag but
; does not abort - the install then proceeds exactly as it does today and the orphan
; simply survives to the next upgrade. Nothing here can fail the install.

!macro NSIS_HOOK_PREINSTALL
  DetailPrint "Removing superseded NVRTC runtime files from sidecar..."
  ; Both patterns are re-extracted from the bundle by the file copy that follows.
  ; Wildcards match files only, so sidecar\models\ is never a candidate.
  Delete "$INSTDIR\sidecar\nvrtc64_*.dll"
  Delete "$INSTDIR\sidecar\nvrtc-builtins64_*.dll"
!macroend
