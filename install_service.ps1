# =============================================================================
# install_service.ps1 -- Register ubx-ray as a Windows service via NSSM.
#
# Usage (Administrator PowerShell):
#   .\install_service.ps1                          # default port 8000
#   .\install_service.ps1 -Port 8080
#   .\install_service.ps1 -AdminToken "secret"
#   .\install_service.ps1 -Uninstall               # remove service + firewall
#
# What it does:
#   1. Verifies admin rights and the .venv layout.
#   2. Downloads NSSM into .\tools\nssm\ (if not already present).
#   3. Registers a "ubxray" Windows service that runs:
#        .\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port <Port>
#      under the LocalSystem account, AutoStart at boot, with stdout/stderr
#      logs rotated at 10 MB into .\logs\.
#   4. Adds an inbound firewall rule for the listening TCP port.
#   5. Starts the service.
#
# Re-running the script is safe: it stops the existing service, updates the
# config, and starts it again.
# =============================================================================

[CmdletBinding()]
param(
    [int]    $Port           = 8000,
    [string] $AdminToken     = "",
    [int]    $MaxUploadMB    = 300,
    [string] $ServiceName    = "ubxray",
    [switch] $Uninstall
)

$ErrorActionPreference = "Stop"

# ---------- helpers ---------------------------------------------------------

function Write-Step($msg)  { Write-Host "[*] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[+] $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[x] $msg" -ForegroundColor Red }

function Assert-Admin {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Err "This script must be run as Administrator."
        Write-Host "    Right-click PowerShell -> 'Run as administrator', then re-run." -ForegroundColor Yellow
        exit 1
    }
}

# ---------- paths -----------------------------------------------------------

$ProjectRoot  = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython   = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogsDir      = Join-Path $ProjectRoot "logs"
$ToolsDir     = Join-Path $ProjectRoot "tools"
$NssmDir      = Join-Path $ToolsDir   "nssm"
$NssmExe      = Join-Path $NssmDir    "nssm.exe"

Assert-Admin

# ---------- uninstall path -------------------------------------------------

if ($Uninstall) {
    Write-Step "Uninstalling service '$ServiceName'..."
    if (Test-Path $NssmExe) {
        & $NssmExe stop   $ServiceName 2>$null | Out-Null
        & $NssmExe remove $ServiceName confirm 2>$null | Out-Null
        Write-Ok "Service removed."
    } else {
        # Fall back to sc.exe in case NSSM has been deleted.
        sc.exe stop   $ServiceName 2>$null | Out-Null
        sc.exe delete $ServiceName 2>$null | Out-Null
        Write-Ok "Service removed (via sc.exe)."
    }

    $rule = "ubx-ray TCP $Port"
    if (Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue) {
        Remove-NetFirewallRule -DisplayName $rule
        Write-Ok "Firewall rule '$rule' removed."
    }
    Write-Ok "Uninstall done."
    exit 0
}

# ---------- preflight -------------------------------------------------------

Write-Step "Project root: $ProjectRoot"

if (-not (Test-Path $VenvPython)) {
    Write-Err ".venv not found at $VenvPython"
    Write-Host "    Create it first:" -ForegroundColor Yellow
    Write-Host "      python -m venv .venv" -ForegroundColor Yellow
    Write-Host "      .\.venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host "      pip install fastapi uvicorn jinja2 python-multipart" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path (Join-Path $ProjectRoot "app.py"))) {
    Write-Err "app.py not found in $ProjectRoot."
    exit 1
}

if (-not (Test-Path $LogsDir))  { New-Item -ItemType Directory -Path $LogsDir  | Out-Null }
if (-not (Test-Path $ToolsDir)) { New-Item -ItemType Directory -Path $ToolsDir | Out-Null }

# ---------- NSSM bootstrap --------------------------------------------------

if (-not (Test-Path $NssmExe)) {
    Write-Step "Downloading NSSM..."

    # nssm.cc occasionally serves 503; fall back to the latest CI build URL
    # and the web.archive.org snapshot before giving up. Each attempt is a
    # plain HTTPS GET of a static zip -- the binary inside is identical
    # across mirrors.
    $zipPath  = Join-Path $ToolsDir "nssm.zip"
    $mirrors = @(
        "https://nssm.cc/release/nssm-2.24.zip",
        "https://nssm.cc/ci/nssm-2.24-101-g897c7ad.zip",
        "https://web.archive.org/web/2024/https://nssm.cc/release/nssm-2.24.zip"
    )

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $downloaded = $false
    foreach ($url in $mirrors) {
        try {
            Write-Host "    -> $url" -ForegroundColor DarkGray
            Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing -TimeoutSec 30
            if ((Get-Item $zipPath).Length -gt 100KB) { $downloaded = $true; break }
        } catch {
            Write-Warn2 "    failed: $($_.Exception.Message)"
        }
    }

    if (-not $downloaded) {
        Write-Err "Could not download NSSM from any mirror."
        Write-Host ""
        Write-Host "Manual workaround:" -ForegroundColor Yellow
        Write-Host "  1. Download nssm-2.24.zip from https://nssm.cc/download" -ForegroundColor Yellow
        Write-Host "     (or 'choco install nssm' if Chocolatey is available)" -ForegroundColor Yellow
        Write-Host "  2. Extract win64\nssm.exe to:" -ForegroundColor Yellow
        Write-Host "     $NssmExe" -ForegroundColor Yellow
        Write-Host "  3. Re-run this script -- the download step will be skipped." -ForegroundColor Yellow
        exit 1
    }

    Write-Step "Extracting NSSM..."
    Expand-Archive -Path $zipPath -DestinationPath $ToolsDir -Force

    # Pick win64 binary out of the extracted tree -> .\tools\nssm\nssm.exe
    $extracted = Get-ChildItem $ToolsDir -Directory -Filter "nssm-*" | Select-Object -First 1
    if (-not $extracted) {
        Write-Err "NSSM zip layout unexpected -- could not find nssm-*\ folder."
        exit 1
    }
    $arch = if ([Environment]::Is64BitOperatingSystem) { "win64" } else { "win32" }
    $srcExe = Join-Path $extracted.FullName "$arch\nssm.exe"
    if (-not (Test-Path $srcExe)) {
        Write-Err "Could not find $srcExe in NSSM archive."
        exit 1
    }
    if (-not (Test-Path $NssmDir)) { New-Item -ItemType Directory -Path $NssmDir | Out-Null }
    Copy-Item $srcExe $NssmExe -Force
    Remove-Item $zipPath -Force
    Remove-Item $extracted.FullName -Recurse -Force
    Write-Ok "NSSM ready at $NssmExe"
} else {
    Write-Ok "NSSM already present: $NssmExe"
}

# ---------- service registration --------------------------------------------

# If the service already exists, stop and remove it first so NSSM `install`
# starts from a clean slate. Re-running the script should be idempotent.
$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Step "Existing '$ServiceName' service found -- stopping and removing..."
    & $NssmExe stop   $ServiceName 2>$null | Out-Null
    & $NssmExe remove $ServiceName confirm | Out-Null
    Start-Sleep -Seconds 1
}

Write-Step "Installing service '$ServiceName'..."
$uvicornArgs = "-m uvicorn app:app --host 0.0.0.0 --port $Port"
& $NssmExe install $ServiceName $VenvPython $uvicornArgs | Out-Null

# Working directory + display info
& $NssmExe set $ServiceName AppDirectory  $ProjectRoot              | Out-Null
& $NssmExe set $ServiceName DisplayName   "ubX-ray FastAPI Server"  | Out-Null
& $NssmExe set $ServiceName Description   "ubx-ray UBX log analysis web service (uvicorn + FastAPI)" | Out-Null
& $NssmExe set $ServiceName Start         SERVICE_AUTO_START        | Out-Null
& $NssmExe set $ServiceName ObjectName    LocalSystem               | Out-Null

# Logging
$outLog = Join-Path $LogsDir "ubxray.out.log"
$errLog = Join-Path $LogsDir "ubxray.err.log"
& $NssmExe set $ServiceName AppStdout       $outLog     | Out-Null
& $NssmExe set $ServiceName AppStderr       $errLog     | Out-Null
& $NssmExe set $ServiceName AppRotateFiles  1           | Out-Null
& $NssmExe set $ServiceName AppRotateOnline 1           | Out-Null
& $NssmExe set $ServiceName AppRotateBytes  10485760    | Out-Null  # 10 MB

# Restart policy: NSSM default is to restart immediately and keep restarting.
# Make crash recovery a bit gentler (3s -> 30s ceiling).
& $NssmExe set $ServiceName AppExit        Default Restart | Out-Null
& $NssmExe set $ServiceName AppRestartDelay 3000             | Out-Null
& $NssmExe set $ServiceName AppThrottle     5000             | Out-Null

# Environment variables. NSSM AppEnvironmentExtra takes pairs joined by NUL,
# but PowerShell 5.1 doesn't accept `0 in a string. Pass them as repeated
# `KEY=value` arguments instead -- NSSM accepts that form too.
$envArgs = @()
$envArgs += "PYTHONUNBUFFERED=1"
$envArgs += "UBXRAY_MAX_UPLOAD_MB=$MaxUploadMB"
if ($AdminToken -ne "") {
    $envArgs += "UBXRAY_ADMIN_TOKEN=$AdminToken"
}
& $NssmExe set $ServiceName AppEnvironmentExtra @envArgs | Out-Null

Write-Ok "Service '$ServiceName' configured."

# ---------- firewall --------------------------------------------------------

$ruleName = "ubx-ray TCP $Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    Write-Step "Adding firewall rule '$ruleName' (inbound TCP $Port)..."
    New-NetFirewallRule -DisplayName $ruleName `
        -Direction Inbound -Protocol TCP -LocalPort $Port `
        -Action Allow -Profile Any | Out-Null
    Write-Ok "Firewall rule added."
} else {
    Write-Ok "Firewall rule '$ruleName' already present."
}

# ---------- start -----------------------------------------------------------

Write-Step "Starting service..."
& $NssmExe start $ServiceName | Out-Null

# NSSM has a 5 s throttle (AppThrottle) before the SCM marks the service as
# Running, so an immediate Get-Service usually reports StartPending. Poll for
# up to 15 s before giving up.
$deadline = (Get-Date).AddSeconds(15)
do {
    Start-Sleep -Milliseconds 500
    $svc = Get-Service -Name $ServiceName
} while ($svc.Status -ne "Running" -and (Get-Date) -lt $deadline)

if ($svc.Status -ne "Running") {
    Write-Warn2 "Service status: $($svc.Status) after 15 s. Tail of err log:"
    if (Test-Path $errLog) {
        Get-Content $errLog -Tail 30 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
    }
    Write-Host "    Full log: $errLog" -ForegroundColor Yellow
    exit 1
}

# Local IPv4 hint for the user
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
       Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*" -and $_.PrefixOrigin -ne "WellKnown" } |
       Select-Object -First 1).IPAddress

Write-Host ""
Write-Ok "ubx-ray service is running."
Write-Host "    Local:    http://localhost:$Port" -ForegroundColor White
if ($ip) {
    Write-Host "    LAN:      http://$ip`:$Port"  -ForegroundColor White
}
Write-Host "    Logs:     $LogsDir"               -ForegroundColor White
Write-Host ""
Write-Host "Common operations:" -ForegroundColor Gray
Write-Host "    Restart:  Restart-Service $ServiceName"     -ForegroundColor Gray
Write-Host "    Stop:     Stop-Service    $ServiceName"     -ForegroundColor Gray
Write-Host "    Status:   Get-Service     $ServiceName"     -ForegroundColor Gray
Write-Host "    Edit:     .\tools\nssm\nssm.exe edit $ServiceName"  -ForegroundColor Gray
Write-Host "    Remove:   .\install_service.ps1 -Uninstall" -ForegroundColor Gray
