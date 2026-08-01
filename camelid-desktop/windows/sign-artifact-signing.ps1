# Sign one file with Azure Artifact Signing (formerly Trusted Signing).
#
# This is Tauri's `bundle.windows.signCommand` hook: the bundler calls it with the path to
# each binary it is about to package, AFTER it has finished rewriting that binary and BEFORE
# it seals it into the installer. That timing is the whole point.
#
# WHY THIS EXISTS. Tauri stamps the bundle type into the binary it stages for the installer,
# rewriting the marker `__TAURI_BUNDLE_TYPE_VAR_UNK` to `..._NSS` so the installed app knows
# it came from NSIS. It patches a STAGED COPY, not target/release. v0.4.7 signed the binary
# before `tauri bundle` ran, so that rewrite landed on already-signed bytes and every
# installed copy reported HashMismatch -- a broken signature, which is a worse posture than
# the unsigned binary v0.4.6 shipped. Signing from inside this hook is the only point at
# which the bytes are final. Measured on the v0.4.7 artifacts: the installed and portable
# copies differ in exactly two places, that 3-byte marker and the PE CheckSum (which
# Authenticode excludes).
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
