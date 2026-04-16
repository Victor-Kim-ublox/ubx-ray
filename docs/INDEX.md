# ubX-ray Project Documentation Index

A web service for analyzing u-blox UBX binary logs and comparing NMEA position data.

---

## Project Structure

```
ubx-ray/
├── app.py                        ← FastAPI main server
├── ubx2kmz.py                    ← UBX → KMZ conversion script
├── nmea_comparison.py            ← NMEA position error analysis module
├── templates/
│   ├── home.html                 ← Home screen (3 tabs: Single / Multi / NMEA)
│   ├── compare4.html             ← Multi upload standalone page
│   ├── compare4_report.html      ← Multi comparison report
│   ├── compare4_view.html        ← Split map view
│   ├── compare4_overlay.html     ← Overlay map view
│   ├── map.html                  ← Single file map viewer
│   ├── report.html               ← Single file analysis report
│   ├── report_nmea.html          ← NMEA comparison report
│   └── recent.html               ← Recent results list
├── static/
│   └── favicon.png
├── data/
│   └── ubxray.sqlite3            ← SQLite database (WAL mode)
├── uploads/                      ← Original UBX files + graph JSON
├── outputs/{rid}/                ← result.kmz
└── docs/                         ← This directory
    ├── INDEX.md                  ← Full index (this file)
    ├── app.md
    ├── ubx2kmz.md
    ├── nmea_comparison.md
    ├── template_home.md
    ├── template_compare4.md
    ├── template_compare4_report.md
    ├── template_compare4_view.md
    ├── template_compare4_overlay.md
    ├── template_map.md
    ├── template_report.md
    ├── template_report_nmea.md
    └── template_recent.md
```

---

## File Documentation

| File | Docs | Summary |
|---|---|---|
| `app.py` | [app.md](app.md) | FastAPI server: upload, conversion queue, API routes, cleanup daemon |
| `ubx2kmz.py` | [ubx2kmz.md](ubx2kmz.md) | UBX binary → KMZ conversion: NAV-PVT / NAV2 / AID-MAPM parsing |
| `nmea_comparison.py` | [nmea_comparison.md](nmea_comparison.md) | Reference vs. Test NMEA position error (CEP) analysis |
| `templates/home.html` | [template_home.md](template_home.md) | Main home: Single / Multi / NMEA 3-tab SPA |
| `templates/compare4.html` | [template_compare4.md](template_compare4.md) | Multi upload standalone page (`/compare4`) |
| `templates/compare4_report.html` | [template_compare4_report.md](template_compare4_report.md) | Multi comparison report: real-time polling + Chart.js |
| `templates/compare4_view.html` | [template_compare4_view.md](template_compare4_view.md) | 2×2 split map view: OpenLayers, synchronized views |
| `templates/compare4_overlay.html` | [template_compare4_overlay.md](template_compare4_overlay.md) | Single map with 4 overlaid tracks and layer toggle |
| `templates/map.html` | [template_map.md](template_map.md) | Single file map viewer: playback, size control, popups |
| `templates/report.html` | [template_report.md](template_report.md) | Single file analysis report: stats + charts |
| `templates/report_nmea.html` | [template_report_nmea.md](template_report_nmea.md) | NMEA comparison result: CEP stats + error chart |
| `templates/recent.html` | [template_recent.md](template_recent.md) | Recent results list: status badges + download links |

---

## Core Data Flow

### Single UBX Analysis
```
[Home Single tab] → POST /upload
  → quick_ubx_summary()        # fast epoch count (mmap)
  → DB INSERT (status=queued)
  → enqueue_convert()          # async queue
    → run_ubx2kmz()            # runs ubx2kmz.py as subprocess
      → generates KMZ, saves graph JSON
      → DB UPDATE (status=done, epoch_total, epoch_missing)
  → Redirect /report/{rid}
    → report.html renders (polls until done, then shows charts + map link)
```

### Multi UBX Comparison
```
[Home Multi tab or /compare4] → POST /compare4/upload
  → DB INSERT + enqueue_convert() for each file (parallel)
  → Redirect /compare4/report/{r1}/{r2}/{r3}/{r4}
    → compare4_report.html (2s polling → Chart.js charts when done)
    → [Split Map View]    /compare4/view/...    → compare4_view.html
    → [Overlay Map View]  /compare4/overlay/... → compare4_overlay.html
```

### NMEA Comparison
```
[Home NMEA tab] → POST /analyze_nmea
  → save files → run_in_threadpool(analyze_nmea_files)
  → render report_nmea.html directly (not persisted to DB)
```

---

## API Endpoint Summary

| Endpoint | Description |
|---|---|
| `GET /api/status/{rid}` | Poll conversion status (status, has_kmz, error) |
| `GET /api/graph/{rid}` | Return graph JSON (chart data) |
| `GET /kml/{rid}` | Extract and return doc.kml from KMZ |
| `GET /download?path=` | Download KMZ file (ownership verified) |

---

## Tech Stack

| Area | Technology |
|---|---|
| Web framework | FastAPI |
| Templating | Jinja2 |
| Database | SQLite (WAL, per-request connection) |
| Map rendering | OpenLayers 9.1.0 |
| Charts | Chart.js 4.4.0 + chartjs-plugin-zoom |
| Binary parsing | Python mmap + memoryview |
| Conversion | subprocess (ubx2kmz.py) |
| Async execution | asyncio + ThreadPoolExecutor |
| User identification | HTTP cookie (anonymous UUID) |
