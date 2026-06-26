# ubX-ray

A web application for analyzing **u-blox UBX** GNSS binary logs. Upload a
`.ubx` / `.bin` file and get an interactive map track, accuracy / fix-type /
signal-strength charts, jamming & spoofing (SEC-SIG) overlays, and a
downloadable KMZ — plus 4-way comparison of multiple logs side by side.

Built with FastAPI + Jinja2 (no build step), OpenLayers for maps, and
Chart.js for graphs.

---

## Features

- **Single-file analysis** — NAV-PVT track on an OpenLayers map (Road /
  Satellite), playback with speed control, distance measurement, and a KMZ
  export.
- **Charts** — 2D/3D accuracy, fix type, CN0 (top-5 avg, qualityInd-filtered),
  and UBX-SEC-SIG jamming/spoofing state, all zoom/pan-synced on a shared time
  axis.
- **4-way comparison** — split map view (one panel per file), overlay map view
  (all tracks on one map), and a comparison report (summary table, accuracy
  overlay, CDF, fix-type distribution, speed/altitude/SV, CNO).
- **NMEA comparison** — reference vs. test GGA position-error analysis
  (Haversine, CEP50/CEP95).
- **Safe by default** — per-user isolation via anonymous cookie, upload size
  limit, UBX magic-byte validation, rate limiting, and a background cleanup
  daemon (TTL / quota / per-user caps).

---

## Quick Start (development)

Requires Python 3.10+.

```powershell
# 1. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install fastapi "uvicorn[standard]" jinja2 python-multipart
#   (optional) pip install pyubx2

# 3. Run the dev server with auto-reload
uvicorn app:app --reload
```

Then open <http://localhost:8000>.

To expose it on the LAN:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

> If PowerShell blocks script activation, run once:
> `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Running as a Windows Service (production)

For an always-on server that starts at boot **without** anyone logging in,
use the NSSM-based installer. Run from an **Administrator** PowerShell:

```powershell
.\install_service.ps1                       # default port 8000
.\install_service.ps1 -Port 8080 -AdminToken "secret" -MaxUploadMB 1024
```

The script downloads NSSM, registers the `ubxray` service (LocalSystem,
auto-start, log rotation, firewall rule) and starts it. `--reload` is
intentionally omitted in production — restart explicitly after code changes.

See [`docs/install_service.md`](docs/install_service.md) for full details.

### Service operations

```powershell
# Status / control
Get-Service     ubxray
Restart-Service ubxray            # apply code changes
Stop-Service    ubxray
Start-Service   ubxray

# Tail the logs
Get-Content .\logs\ubxray.out.log -Wait -Tail 50
Get-Content .\logs\ubxray.err.log -Wait -Tail 50

# Edit configuration (NSSM GUI)
.\tools\nssm\nssm.exe edit ubxray

# Uninstall (service + firewall rule)
.\install_service.ps1 -Uninstall
```

---

## Configuration

| Env Var | Default | Purpose |
|---|---|---|
| `UBXRAY_MAX_UPLOAD_MB` | `1024` | Max upload file size (MB) |
| `UBXRAY_MAX_TOTAL_GB` | `100` | Total disk budget for uploads + outputs; LRU cleanup above this |
| `UBXRAY_ADMIN_TOKEN` | _(none)_ | Bearer token to bypass per-user ownership checks |

Other limits in `app.py`: 10 results per user, 7-day TTL, cleanup every
3600 s, `MAX_CONVERT` = CPU cores ÷ 2. At the 1 GB upload limit, one user can
use ~10 GB (10 results), so the 100 GB quota holds ~10 users at full capacity
before LRU cleanup kicks in — raise `UBXRAY_MAX_TOTAL_GB` if you expect more.

---

## Admin CLI

`admin_cli.py` lists and deletes stored uploads directly from the server shell
(dry-run by default, `--yes` to commit):

```powershell
python admin_cli.py list
python admin_cli.py show <rid>
python admin_cli.py purge --older-than 7 --yes
```

See [`docs/admin_cli.md`](docs/admin_cli.md).

---

## Project Layout

```
app.py                  FastAPI backend: routes, DB, upload validation,
                        rate limiting, cleanup daemon, concurrency control
ubx2kmz.py              UBX binary parser + KMZ / graph-JSON generator
nmea_comparison.py      NMEA GGA position-error analysis
admin_cli.py            Server-side data management CLI
install_service.ps1     Windows service installer (NSSM)
templates/              Jinja2 HTML templates (home, report, map, compare4, ...)
docs/                   One .md per source file describing its behavior
data/                   SQLite DB (auto-created, gitignored)
uploads/  outputs/      Runtime upload + conversion artifacts (gitignored)
```

### Data flow

```
Upload → quick_ubx_summary() (fast mmap frame count)
       → DB INSERT (status=queued)
       → enqueue_convert() → ubx2kmz → KMZ + graph JSON in outputs/{rid}/
       → DB UPDATE (done/error)
       → /report/{rid} polls /api/status/{rid} until done
```

---

## Notes

- No frontend build toolchain — OpenLayers 9.1.0 and Chart.js 4.4.0 are loaded
  from CDN.
- Each source file has a matching `docs/<name>.md`; update it when changing
  behavior.
- All code comments and documentation are in English.
