# app.py

## Overview
The **main backend** of the ubX-ray web service. An async HTTP server built on FastAPI that handles UBX file upload, conversion, retrieval, and NMEA comparison analysis.

---

## Tech Stack
- **FastAPI** + **Jinja2 Templates** (HTML rendering)
- **SQLite** (WAL mode, new connection per request)
- **ThreadPoolExecutor** + `asyncio.Semaphore` (conversion concurrency control)
- **subprocess** (calls ubx2kmz.py)
- **mmap** (high-speed UBX file scanning)

---

## Key Configuration (env vars / hardcoded)

| Constant | Default | Description |
|---|---|---|
| `MAX_UPLOAD_BYTES` | 1 GB (1024 MB) | Configurable via env var `UBXRAY_MAX_UPLOAD_MB` |
| `CLN_MAX_TOTAL_BYTES` | 100 GB | LRU deletion when disk limit is exceeded; configurable via env var `UBXRAY_MAX_TOTAL_GB` |
| `CLN_MAX_RESULTS_PER_USER` | 10 | Maximum number of results retained per user |
| `CLN_RETAIN_DAYS` | 7 | TTL after upload (days) |
| `CLN_INTERVAL_SEC` | 3600 | Cleanup interval (seconds) |
| `ADMIN_TOKEN` | `""` | Set via `UBXRAY_ADMIN_TOKEN` env var; Bearer auth |
| `COOKIE_NAME` | `ubx_user` | Anonymous user identification cookie |

---

## Directory Structure

```
BASE_DIR/
├── data/ubxray.sqlite3   ← Database
├── uploads/              ← Original UBX files + graph JSON
└── outputs/{rid}/        ← result.kmz output
```

---

## Middleware

### `assign_user_cookie`
Checks the `ubx_user` cookie on every incoming HTTP request. If absent, issues an anonymous ID via `uuid4().hex` and sets a 1-year cookie. Stored in `request.state.user_id` for use by route handlers.

---

## Database Schema (`results` table)

| Column | Type | Description |
|---|---|---|
| `id` | TEXT PK | 8-character hex (timestamp-based or uuid) |
| `user_id` | TEXT | Cookie-based anonymous user ID |
| `filename` | TEXT | Original filename |
| `uploaded_at` | TEXT | ISO8601 UTC timestamp |
| `epoch_total` | INTEGER | Number of NAV-PVT frames |
| `epoch_missing` | INTEGER | Missing epoch count (updated after conversion) |
| `crc_errors` | INTEGER | CRC error count |
| `kmz_path` | TEXT | Absolute path to the generated KMZ file |
| `opts_json` | TEXT | Conversion options as JSON |
| `status` | TEXT | `queued` → `running` → `done` / `error` |
| `error` | TEXT | Error message on failure |

The `ensure_columns()` function automatically adds missing columns to legacy databases.

---

## API Route List

### Single File Analysis

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Home screen (home.html) |
| `POST` | `/upload` | Upload UBX file → register in DB → add to conversion queue → redirect to `/report/{rid}` |
| `POST` | `/upload/chunk` | Append one chunk of a large upload to a per-session `.part` file (see Chunked Upload) |
| `POST` | `/upload/complete` | Assemble a chunked upload and run the normal upload pipeline → redirect to `/report/{rid}` |
| `GET` | `/report/{rid}` | Single file analysis report (report.html) |
| `GET` | `/map/{rid}` | KMZ-based map viewer (map.html) |
| `GET` | `/kml/{rid}` | Extract and return doc.kml from KMZ |
| `GET` | `/download` | Download KMZ file (ownership verified) |
| `GET` | `/recent` | Recent results list (recent.html) |

### Multi File Comparison (compare4)

| Method | Path | Description |
|---|---|---|
| `GET` | `/compare4` | Multi upload page (compare4.html) |
| `POST` | `/compare4/upload` | Upload 1–4 UBX files → queue each → redirect to `/compare4/report/{r1}/{r2}/{r3}/{r4}` |
| `POST` | `/compare4/kml/upload` | Upload 1–4 KML/KMZ tracks (no conversion) → store as `done` → redirect to `/compare4/overlay/{...}` |
| `GET` | `/compare4/report/{r1}/{r2}/{r3}/{r4}` | Analysis report (compare4_report.html) |
| `GET` | `/compare4/view/{r1}/{r2}/{r3}/{r4}` | Split map view (compare4_view.html); hides the Report button for KML groups |
| `GET` | `/compare4/overlay/{r1}/{r2}/{r3}/{r4}` | Overlay map view (compare4_overlay.html); hides the Report button for KML groups |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/status/{rid}` | Poll conversion status (`status`, `has_kmz`, `error`, `filename`; plus `progress` 0–100 while `status='running'`, read from the converter's `.progress` sidecar) |
| `GET` | `/api/graph/{rid}` | Return graph JSON (used for chart rendering in compare4_report) |

### NMEA Comparison

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze_nmea` | Upload 2 NMEA files → analyze → render report_nmea.html |

---

## Chunked Upload

Cloudflare's free plan caps a single request body at ~100 MB, so uploads
through the ubx-ray.com tunnel fail with a **Cloudflare 413** for larger logs
(the app's own limit is `MAX_UPLOAD_MB` = 1024). The home screen therefore
splits files above ~95 MB into 64 MB chunks:

1. `POST /upload/chunk` (sequential, one at a time) — fields `upload_id`
   (client-generated UUID, validated path-safe), `index`, `chunk` (blob).
   Chunks are appended to `uploads/parts/{user_id}_{upload_id}.part`; the
   part file is keyed by the caller's cookie `user_id`, so sessions cannot
   cross users. `index == 0` consumes **one** rate-limit slot (not one per
   chunk) and resets any stale part with the same id; the cumulative size is
   enforced against `MAX_UPLOAD_BYTES` during append.
2. `POST /upload/complete` — fields `upload_id`, `filename`, plus the same
   conversion options as `/upload`. Validates the extension, moves the part to
   `uploads/{rid}_{name}` via `os.replace`, then calls `_finalize_upload()` —
   the shared tail of `/upload` (UBX sniff → quick summary → DB insert →
   enqueue → 303 to `/report/{rid}`).

Abandoned `.part` files (client navigated away mid-upload) are swept by the
cleanup loop after `PART_MAX_AGE_SEC` (24 h). `cln_orphans` skips the
`uploads/parts/` directory (its name contains no underscore, so the rid-prefix
scan ignores it).

Multi-file comparison uploads (`/compare4/upload`) still use a single request
and remain subject to the tunnel's 100 MB cap; use the LAN address for large
multi-file comparisons.

---

## Key Functions

### `looks_like_ubx(path, window=65536)`
Upload-time validator. Returns `True` when the UBX sync pattern `0xB5 0x62` appears anywhere in the first `window` bytes of the saved file (default 64 KB). The sync bytes are not required at offset 0, because many u-blox loggers prepend NMEA sentences or vendor headers before the first UBX frame. Called from both `/upload` and `/compare4/upload`; on `False` the file is deleted and a 400 "Invalid file format" response is returned.

### `extract_kml_bytes(path)` / `write_normalized_kmz(kml_bytes, dest)`
KML-comparison upload helpers. `extract_kml_bytes` returns the KML document
bytes from a `.kml` (raw XML, sniffed for a `<kml` tag) or `.kmz`/zip (prefers
`doc.kml`, else the first `*.kml` entry), or `None` if the file isn't KML.
`write_normalized_kmz` repackages those bytes into a single-entry `doc.kml`
KMZ at `outputs/{rid}/result.kmz`, so the existing `/kml/{rid}` endpoint and
the split/overlay map views serve KML uploads with no special-casing.
`_opts_is_kml(opts_json)` reports whether a result row was a direct KML upload
(used by the map views to hide the Report button).

### `quick_ubx_summary(filepath)`
High-speed scan of the UBX binary via mmap, counting **NAV-PVT frame occurrences**. No CRC verification — only checks the header (sync bytes + class/id + length), making it very fast. Used for the epoch count displayed immediately after upload.

### `run_ubx2kmz(filepath, rid, **opts)`
Runs `ubx2kmz.py` as a subprocess. After completion, copies the auto-generated KMZ file to `outputs/{rid}/result.kmz`, reads stats (epoch_total, epoch_missing) from the co-located `_graph.json`, and updates the database. On timeout (default 1800s) or exception, sets DB status to `error`.

Passes `--progress-file {upload}.progress` so the converter reports its scan
percentage (~0.5 s interval) while running; `/api/status/{rid}` reads that
sidecar and returns it as `progress` (0–100, only while `status='running'`),
which drives the real progress bar on the home screen. The sidecar is removed
in a `finally` block whether the conversion succeeds or fails.

### `_recover_interrupted_jobs()` (startup hook)
A server restart kills in-flight conversions but used to leave their DB rows
stuck in `queued`/`running` forever (the UIs then poll indefinitely). On
startup, every such row is **re-enqueued** when its upload file still exists
(the stored `opts_json` restores the conversion options); rows whose file is
gone (or direct-KML rows) are marked `status='error'` /
`"interrupted by server restart"` so pollers terminate cleanly.

### `enqueue_convert(filepath, rid, **opts)`
Async conversion queue wrapper. Uses `asyncio.Semaphore(MAX_CONVERT)` to cap concurrent conversions at half the number of CPU cores.

### `ensure_owner_or_404(rid, user_id, admin)`
Checks the owner of a rid in the database and returns whether access is permitted along with the HTTP status code. Admin token grants access to all results.

---

## Cleanup System

A background daemon thread (`cln_loop`) starts at app startup and runs cleanup in the following order every `CLN_INTERVAL_SEC` (1 hour):

1. **TTL Expiry** (`cln_expired`): Delete results older than `CLN_RETAIN_DAYS` days
2. **Keep Latest N Per User** (`cln_keep_latest_per_user`): Remove older results beyond the per-user limit
3. **Disk Quota** (`cln_quota`): LRU deletion of oldest completed results when exceeding `CLN_MAX_TOTAL_BYTES` (100 GB default)
4. **Orphan Cleanup** (`cln_orphans`): Remove files/folders not in DB + NULL out kmz_path for missing KMZ files
5. **VACUUM**: Optimize the SQLite file

---

## Conversion Options (upload form parameters)

| Parameter | Type | Description |
|---|---|---|
| `hz` | int / "" | Downsampling Hz (1/2/5/10) |
| `nav2` | bool | Use NAV2-PVT messages |
| `alt_abs` | bool | Set altitudeMode=absolute |
| `ck` | bool | Enable UBX checksum verification |
| `mapm` | bool | Extract AID-MAPM points only |
| `html` | bool | Include HTML output (currently unused) |

---

## File Dependency Map

```
app.py
├── ubx2kmz.py          (invoked via subprocess)
├── nmea_comparison.py  (imported, run via run_in_threadpool)
├── templates/          (rendered via Jinja2)
│   ├── home.html
│   ├── report.html
│   ├── map.html
│   ├── recent.html
│   ├── report_nmea.html
│   ├── compare4.html
│   ├── compare4_report.html
│   ├── compare4_view.html
│   └── compare4_overlay.html
├── uploads/            (original files + graph JSON)
├── outputs/{rid}/      (result.kmz)
└── data/ubxray.sqlite3 (database)
```

---

## UI Theme

All templates use a unified **u-blox branded light theme** based on the u-blox corporate design system.

| Token | Value | Usage |
|---|---|---|
| Primary | `#FF6E59` | Buttons, active tabs, links, brand text |
| Primary Dark | `#E04A35` | Hover states |
| Background | `#F7F7F7` | Page background |
| Card | `#FFFFFF` | Card/panel backgrounds |
| Text | `#1A1A1A` | Headings, primary text |
| Text Body | `#676767` | Body text |
| Muted | `#999999` | Secondary/hint text |
| Border | `#E6E6E6` | Card borders, dividers |
| Error | `#E3140D` | Error states |
| Success | `#557555` | Success states |
| Font | Space Grotesk | Primary typeface (Google Fonts) |

Track/chart data colors (`--c1` through `--c4`) are kept as functional visualization colors and are independent of the theme.
