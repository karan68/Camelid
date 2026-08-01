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

$signtool = $env:CAMELID_SIGNTOOL
$dlib = $env:CAMELID_SIGN_DLIB
$metadata = $env:CAMELID_SIGN_METADATA

if (-not $signtool -and -not $dlib -and -not $metadata) {
    Write-Host "sign-artifact-signing: signing not configured (CAMELID_SIGNTOOL unset) - skipping $Path"
    exit 0
}

# A PARTIAL configuration is a broken release job, not a developer build: fail rather than
# silently skipping and letting the bundler seal an unsigned binary.
foreach ($pair in @(
        @{ Name = 'CAMELID_SIGNTOOL'; Value = $signtool },
        @{ Name = 'CAMELID_SIGN_DLIB'; Value = $dlib },
        @{ Name = 'CAMELID_SIGN_METADATA'; Value = $metadata })) {
    if (-not $pair.Value) {
        throw "signing is partially configured: $($pair.Name) is empty while others are set"
    }
    if ($pair.Name -ne 'CAMELID_SIGNTOOL' -and -not (Test-Path $pair.Value)) {
        throw "$($pair.Name) points at '$($pair.Value)', which does not exist"
    }
}

if (-not (Test-Path $Path)) {
    throw "nothing to sign at '$Path'"
}

Write-Host "sign-artifact-signing: signing $Path"

# Timestamping is not optional here: Artifact Signing certificates are valid for three days,
# so an untimestamped signature stops verifying almost immediately after release.
& $signtool sign /v /debug /fd SHA256 `
    /tr http://timestamp.acs.microsoft.com /td SHA256 `
    /dlib $dlib /dmdf $metadata `
    $Path
if ($LASTEXITCODE -ne 0) {
    throw "signtool failed with exit code $LASTEXITCODE for $Path"
}

# Fail closed on the result, not just the exit code: a signature that does not verify on the
# machine that just produced it will not verify on a user's machine either.
$status = (Get-AuthenticodeSignature -FilePath $Path).Status
if ($status -ne 'Valid') {
    throw "signtool reported success but $Path verifies as '$status'"
}
Write-Host "sign-artifact-signing: $Path -> Valid"
