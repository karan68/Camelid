# Exercise the native title-bar close path with a real desktop and sidecar.
# The staged app uses an empty model directory and never changes the installed app.
param(
    [Parameter(Mandatory = $true)][string]$DesktopExe,
    [Parameter(Mandatory = $true)][string]$EngineExe
)
$ErrorActionPreference = 'Stop'
$desktopSource = (Resolve-Path -LiteralPath $DesktopExe).Path
$engineSource = (Resolve-Path -LiteralPath $EngineExe).Path
$stage = Join-Path ([IO.Path]::GetTempPath()) ('camelid-close-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'models') | Out-Null
Copy-Item -LiteralPath $desktopSource -Destination (Join-Path $stage 'camelid-desktop.exe')
Copy-Item -LiteralPath $engineSource -Destination (Join-Path $stage 'camelid.exe')

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class CamelidCloseProbe {
    public delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr data);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr data);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr hwnd, System.Text.StringBuilder text, int count);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hwnd, uint message, IntPtr wParam, IntPtr lParam);
    public static IntPtr FindWindow(uint processId) {
        IntPtr found = IntPtr.Zero;
        EnumWindows((hwnd, data) => {
            uint owner;
            GetWindowThreadProcessId(hwnd, out owner);
            var title = new System.Text.StringBuilder(512);
            GetWindowText(hwnd, title, title.Capacity);
            if (owner == processId && title.ToString().Contains("Camelid")) {
                found = hwnd;
                return false;
            }
            return true;
        }, IntPtr.Zero);
        return found;
    }
}
'@

try {
    foreach ($mode in @('ready', 'startup')) {
        $desktop = Start-Process -FilePath (Join-Path $stage 'camelid-desktop.exe') -WindowStyle Hidden -PassThru
        $deadline = [DateTime]::UtcNow.AddSeconds(60)
        $child = $null
        $window = [IntPtr]::Zero
        do {
            $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($desktop.Id) AND Name='camelid.exe'" | Select-Object -First 1
            $window = [CamelidCloseProbe]::FindWindow($desktop.Id)
            if ($child -and $window -ne [IntPtr]::Zero) { break }
            Start-Sleep -Milliseconds 100
        } while ([DateTime]::UtcNow -lt $deadline -and !$desktop.HasExited)
        if (!$child -or $window -eq [IntPtr]::Zero) { throw "${mode}: desktop window or sidecar did not start" }
        if ($mode -eq 'ready') {
            if ($child.CommandLine -notmatch '--addr 127\.0\.0\.1:(\d+)') { throw 'Missing sidecar port' }
            $healthUrl = "http://127.0.0.1:$($Matches[1])/v1/health"
            $healthy = $false
            do {
                try { $healthy = (Invoke-RestMethod $healthUrl -TimeoutSec 1).ok } catch { }
                if (!$healthy) { Start-Sleep -Milliseconds 100 }
            } while (!$healthy -and [DateTime]::UtcNow -lt $deadline)
            if (!$healthy) { throw 'Sidecar never became healthy' }
        }
        # WM_CLOSE is the same native request produced by the title-bar X / Alt+F4.
        if (![CamelidCloseProbe]::PostMessage($window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)) { throw 'WM_CLOSE failed' }
        if (!$desktop.WaitForExit(10000)) { throw "${mode}: desktop survived window close" }
        $deadline = [DateTime]::UtcNow.AddSeconds(5)
        do {
            $remaining = Get-CimInstance Win32_Process -Filter "ProcessId=$($child.ProcessId)"
            if (!$remaining) { break }
            Start-Sleep -Milliseconds 100
        } while ([DateTime]::UtcNow -lt $deadline)
        if ($remaining) { throw "${mode}: sidecar survived desktop exit" }
        Write-Output "PASS ${mode}: title-bar close stopped desktop and sidecar"
    }
} finally {
    # Only processes launched from this unique test directory are eligible for cleanup.
    $testProcesses = @(Get-Process camelid,camelid-desktop -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -in @((Join-Path $stage 'camelid.exe'), (Join-Path $stage 'camelid-desktop.exe')) } |
        ForEach-Object { $_ })
    foreach ($testProcess in $testProcesses) {
        if (!$testProcess.HasExited) { $testProcess.Kill(); $testProcess.WaitForExit() }
    }
    $resolvedStage = [IO.Path]::GetFullPath($stage)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolvedStage.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or
        !(Split-Path $resolvedStage -Leaf).StartsWith('camelid-close-')) { throw 'Unsafe test cleanup path' }
    # WebView subprocesses and antivirus can briefly retain file handles after exit.
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try { Remove-Item -LiteralPath $resolvedStage -Recurse -Force; break }
        catch { if ($attempt -eq 19) { Write-Warning "Test files retained at ${resolvedStage}: $_" } else { Start-Sleep -Milliseconds 100 } }
    }
}
