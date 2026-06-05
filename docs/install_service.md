# install_service.ps1

PowerShell installer that registers ubx-ray as a Windows service via
[NSSM](https://nssm.cc/). Once installed, the FastAPI/uvicorn server
starts automatically at boot **without requiring user logon** and is
auto-restarted by Windows if it crashes.

## Quick start

Open **PowerShell as Administrator**, `cd` into the project root, then:

```powershell
.\install_service.ps1
```

Defaults:

| Setting           | Value                                     |
|-------------------|-------------------------------------------|
| Service name      | `ubxray`                                  |
| Listen address    | `0.0.0.0:8000`                            |
| Account           | `LocalSystem` (no logon required)         |
| Start type        | `Automatic` (boot)                        |
| Crash recovery    | NSSM auto-restart (3 s delay, 5 s throttle) |
| Log location      | `.\logs\ubxray.out.log`, `.\logs\ubxray.err.log` (10 MB rotate) |

## Custom port / admin token

```powershell
.\install_service.ps1 -Port 8080 -AdminToken "mysecret" -MaxUploadMB 500
```

Re-running the script is safe -- it stops the old service, rewrites the
configuration, and starts a fresh copy.

## Uninstall

```powershell
.\install_service.ps1 -Uninstall
```

Removes the service and the firewall rule. NSSM itself stays in
`.\tools\nssm\` so the next install does not need to re-download it.

## What the script does

1. **Admin check** -- aborts if not elevated.
2. **Preflight** -- verifies `.\.venv\Scripts\python.exe` and `.\app.py` exist.
3. **NSSM bootstrap** -- if `.\tools\nssm\nssm.exe` is missing, downloads
   `nssm-2.24.zip`, extracts the win64/win32 binary into `tools\nssm\`,
   and discards the rest of the archive.
4. **Service install** -- registers `ubxray` with:
   - Command:  `.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port <Port>`
   - Working dir:  project root
   - Account:  `LocalSystem`
   - Start:    `SERVICE_AUTO_START`
   - Logs:     `logs\ubxray.{out,err}.log` with 10 MB rotation
   - Env vars: `PYTHONUNBUFFERED=1`, `UBXRAY_MAX_UPLOAD_MB`, `UBXRAY_ADMIN_TOKEN` (if provided)
5. **Firewall** -- inbound rule `"ubx-ray TCP <Port>"` allowing TCP on the
   chosen port (only if not already present).
6. **Start** -- `nssm start ubxray`, then prints the local + LAN URL.

## Operations cheat sheet

```powershell
# State / control
Get-Service     ubxray
Restart-Service ubxray
Stop-Service    ubxray
Start-Service   ubxray

# Edit configuration interactively (NSSM GUI)
.\tools\nssm\nssm.exe edit ubxray

# Tail the log
Get-Content .\logs\ubxray.out.log -Wait -Tail 50
Get-Content .\logs\ubxray.err.log -Wait -Tail 50

# Update environment variables without removing the service
.\tools\nssm\nssm.exe set ubxray AppEnvironmentExtra `
    PYTHONUNBUFFERED=1 UBXRAY_MAX_UPLOAD_MB=300 UBXRAY_ADMIN_TOKEN=newtoken
Restart-Service ubxray
```

## Why NSSM (not Task Scheduler / startup folder)

| Method                      | Auto-login required? | Boot-direct? | Auto-restart on crash? |
|-----------------------------|----------------------|--------------|------------------------|
| **NSSM service** (this)     | No                   | Yes          | Yes                    |
| Task Scheduler `AtStartup`  | No                   | Yes          | Manual policy          |
| Startup folder shortcut     | **Yes**              | No           | No                     |

Since auto-login is disabled on this host, the startup folder approach
won't work -- the server would only launch after a human logs in. NSSM
runs as a true Windows service and is independent of any user session.

## Troubleshooting

- **Service won't start:** check `logs\ubxray.err.log`. The most common
  causes are a missing dependency in `.venv` (`pip install -r ...`) or a
  port already in use (`Get-NetTCPConnection -LocalPort 8000`).
- **Port 8000 not reachable from another machine:** confirm the firewall
  rule (`Get-NetFirewallRule -DisplayName 'ubx-ray TCP 8000'`) and that
  the corporate firewall / router permits the port.
- **NSSM download fails:** download `nssm-2.24.zip` manually from
  <https://nssm.cc/download>, place `nssm.exe` at
  `.\tools\nssm\nssm.exe`, and re-run the script. The bootstrap step is
  skipped when the binary is already present.
