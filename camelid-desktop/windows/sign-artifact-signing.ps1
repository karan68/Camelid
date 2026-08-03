# Sign one file with Azure Artifact Signing (formerly Trusted Signing).
#
# This is Tauri's `bundle.windows.signCommand`. The release workflow GENERATES
# tauri.signing.conf.json pointing at this script with ABSOLUTE paths and passes it as an extra
# `--config`. It is deliberately not wired into the committed tauri.conf.json: the paths are
# machine-specific, and a developer running `tauri build` should not need signing tooling.
#
# WHY THIS EXISTS. Tauri patches the binary with the bundle type
# (`__TAURI_BUNDLE_TYPE_VAR_UNK` -> `..._NSS`) so the installed app knows how it was installed,
# and signs only afterwards. A signature applied before `tauri build` is invalidated by that
# rewrite; one applied after cannot reach a binary already sealed inside the installer. v0.4.6
# shipped that copy NotSigned, v0.4.7 shipped it HashMismatch -- a broken signature, worse than
# none. This hook is the only point at which the bytes are final.
#
# WHY ABSOLUTE PATHS. Tauri calls this hook SEVEN times per bundle -- the app binary, five NSIS
# plugin DLLs, and the uninstaller staged as %TEMP%\nst*.tmp -- and the working directory is NOT
# constant across them. Measured on a real bundle: six run from the Tauri project directory, but
# the uninstaller call runs from target\release\nsis\x64, where a project-relative path does not
# resolve. v0.4.8's release died exactly there -- `failed to bundle project: failed to run
# powershell`, with this script never executing and no Windows installer published.
#
# Being invoked on the plugin DLLs and the uninstaller is correct, not incidental: those are the
# installer's own components and ship inside it.
#
# AUTHENTICATION. signtool reaches the service through Azure.CodeSigning.Dlib, which uses
# DefaultAzureCredential. The release workflow runs `azure/login` with OIDC beforehand, so
# the az CLI session is what this authenticates with -- no client secret is stored anywhere.
# `artifact-signing-cli`, the CLI Tauri's own docs suggest, requires AZURE_CLIENT_SECRET and
# was rejected for exactly that reason.
#
# NOT CONFIGURED = NOT SIGNED, LOUDLY SKIPPED. A developer running `tauri build` on their own
# machine has none of these tools, so the hook no-ops rather than failing their build. That
# is safe only because the release workflow independently proves the shipped installer
# carries a valid signature: it installs the built setup.exe and checks the unpacked binary.
# Without that gate this skip would be a silent way to ship unsigned bytes.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path
)

$ErrorActionPreference = 'Stop'

# EVERYTHING goes to a log file as well as stdout.
#
# Tauri SWALLOWS this script's output when it exits non-zero -- it prints
# `failed to bundle project: failed to run <cmd>` and nothing else. Three consecutive releases
# died here and every diagnosis was guesswork, because the one thing that would have explained
# the failure was the output being discarded. CAMELID_SIGN_LOG is an absolute path the workflow
# sets and prints after the bundle step, whether that step passed or failed.
$logPath = $env:CAMELID_SIGN_LOG
function Note([string]$message) {
    $line = "sign-artifact-signing: $message"
    Write-Host $line
    if ($logPath) {
        try { Add-Content -LiteralPath $logPath -Value "[$(Get-Date -Format o)] $line" } catch { }
    }
}

# NOTHING may die silently in here again.
#
# v0.5.1 signed successfully and then vanished: signtool reported `Number of errors: 0`, and
# 0.28s later Tauri reported `failed to run <cmd>` with the log ending mid-script. An unhandled
# terminating error under `ErrorActionPreference = 'Stop'` produces exactly that -- the process
# exits non-zero, and Tauri discards the stderr that would have named it. This trap makes any
# such error self-reporting instead of another round of inference.
trap {
    Note "FATAL: unhandled error at line $($_.InvocationInfo.ScriptLineNumber): $($_.Exception.GetType().Name): $($_.Exception.Message)"
    exit 1
}

$signtool = $env:CAMELID_SIGNTOOL
$dlib = $env:CAMELID_SIGN_DLIB
$metadata = $env:CAMELID_SIGN_METADATA

if (-not $signtool -and -not $dlib -and -not $metadata) {
    Note "signing not configured (CAMELID_SIGNTOOL unset) - skipping $Path"
    exit 0
}

# A PARTIAL configuration is a broken release job, not a developer build: fail rather than
# silently skipping and letting the bundler seal an unsigned binary.
foreach ($pair in @(
        @{ Name = 'CAMELID_SIGNTOOL'; Value = $signtool },
        @{ Name = 'CAMELID_SIGN_DLIB'; Value = $dlib },
        @{ Name = 'CAMELID_SIGN_METADATA'; Value = $metadata })) {
    if (-not $pair.Value) {
        Note "FATAL: signing is partially configured: $($pair.Name) is empty while others are set"
        exit 1
    }
    if (-not (Test-Path -LiteralPath $pair.Value)) {
        Note "FATAL: $($pair.Name) points at '$($pair.Value)', which does not exist"
        exit 1
    }
}

if (-not (Test-Path -LiteralPath $Path)) {
    Note "FATAL: nothing to sign at '$Path'"
    exit 1
}

Note "signing $Path"

# Timestamping is not optional here: Artifact Signing certificates are valid for three days,
# so an untimestamped signature stops verifying almost immediately after release.
#
# `$ErrorActionPreference` is dropped to Continue for exactly this call. Windows PowerShell wraps
# every stderr line of a native command in a NativeCommandError ErrorRecord when the stream is
# redirected, and under 'Stop' that is TERMINATING -- the script would die on signtool's first
# diagnostic line and never reach the exit-code check below, turning any signing hiccup into an
# unexplained bundle failure. Verified locally: with 'Stop' this path exited 1 instead of 0.
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $output = & $signtool sign /v /debug /fd SHA256 `
        /tr http://timestamp.acs.microsoft.com /td SHA256 `
        /dlib $dlib /dmdf $metadata `
        $Path 2>&1
    $signExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
foreach ($line in @($output)) { Note "  signtool| $line" }

# A SIGNTOOL FAILURE IS NOT FATAL TO THE BUILD.
#
# The bundler must still produce an installer. Failing here is what left users with no Windows
# installer at all on v0.4.8 and v0.5.0 -- strictly worse than the unsigned-payload installer
# v0.4.6 shipped, which at least installed and ran. An unsigned binary is a known, previously
# accepted posture; no binary is not. The release's `Prove the INSTALLED binary is signed` step
# reports what actually shipped, and `verify-release-assets` guarantees an incomplete release
# never becomes `latest`.
if ($signExit -ne 0) {
    Note "WARNING: signtool exited $signExit for $Path - shipping this file UNSIGNED rather than failing the bundle"
    exit 0
}

# Read back what was actually written, with retries.
#
# signtool has just closed this file and the signing service wrote to it moments earlier, so the
# read can lose a race against antivirus or a lingering handle. That read must never be able to
# kill the bundle: on v0.5.1 signtool reported `Number of errors: 0` and the script still exited
# non-zero 0.28s later, with the log ending right here -- the signature: an unhandled terminating
# error in this exact read.
$status = $null
for ($attempt = 1; $attempt -le 5; $attempt++) {
    try {
        $status = (Get-AuthenticodeSignature -LiteralPath $Path).Status
        break
    } catch {
        Note "  verify attempt $attempt could not read the signature: $($_.Exception.Message)"
        Start-Sleep -Milliseconds 300
    }
}

# UNREADABLE is not the same as BROKEN, and must not be treated as such.
#
# Failing here would throw away a correctly signed binary over a transient file lock, and put us
# straight back to the no-installer outcome of v0.4.8 and v0.5.0. The release's `Prove the
# INSTALLED binary is signed` step opens the finished installer and checks the payload for real;
# that is the authoritative verdict, and it runs on a settled file.
if ($null -eq $status) {
    Note "WARNING: could not read back the signature of $Path after signing - continuing; the release install-verify gate is authoritative"
    exit 0
}

# A CONFIRMED bad signature is the one fatal case.
#
# signtool claiming success while the result verifies as something other than Valid means we are
# about to seal a BROKEN signature into the installer -- exactly what v0.4.7 shipped, where every
# installed copy reported HashMismatch. Windows reports that as a tampered chain, it earns no
# SmartScreen reputation, and it cannot be explained away as "unsigned".
if ($status -ne 'Valid') {
    Note "FATAL: signtool reported success but $Path verifies as '$status' - refusing to seal a broken signature"
    exit 1
}
Note "$Path -> Valid"
