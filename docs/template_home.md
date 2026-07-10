# templates/home.html

## Overview
The **main home screen** of ubX-ray. Provides five analysis panes
(Single / Multi / KML Comparison / NMEA / Reference vs DUT) in a single-page
SPA-style interface. Four have tab buttons in the nav (Single, Multi, KML
Comparison, Reference vs DUT); the NMEA pane is reachable via `?tab=nmea`.

---

## URL
`GET /`

---

## Tab Structure

### Tab 1: Single .ubx Analysis (`#tab-single`)
Upload and analyze a single UBX/BIN file.

**Form elements:**
- Drag-and-drop + click file selector (`fileInput`, `.ubx/.bin`)
- Conversion Options fieldset
  - `hz`: Downsampling (None / 1 / 2 / 5 / 10 Hz)
  - `nav2`: Use NAV2-PVT checkbox
  - `alt_abs`: altitudeMode=absolute checkbox
  - `mapm`: AID-MAPM only checkbox
- Upload & Analyze button → `POST /upload` (fetch API with JSON redirect handling)

**Footer:** View Recent Results link + hint text

### Tab 2: Multi .ubx Analysis (`#tab-multi`)
Upload 1–4 UBX/BIN files for comparison analysis.

**Form elements:**
- 2×2 grid of dropzones (File 1–4; File 1 is required)
  - `required` / `optional` tags shown in slot labels
  - On file selection: border color changes + opacity transitions to 1.0
- Conversion Options fieldset (same options as Single tab)
- Upload & Compare button → `POST /compare4/upload` (standard form submit)
- Loading text adapts to file count: "Uploading N file(s)... please wait"

**Notice box:** Explains how KMZ background processing and automatic map sync work

### Tab 3: KML Comparison (`#tab-kml`)
Upload 1–4 **KML/KMZ** tracks to compare directly on a shared map (no UBX
conversion). Modeled on the Multi tab's input screen.

**Form elements:**
- 2×2 grid of dropzones (KML 1–4; KML 1 is required), inputs `kFile1..4`,
  dropzones `kDrop1..4`, name slots `kName1..4`, `accept=".kml,.kmz"`
- Reuses `wireMultiDropzone` (border/opacity feedback) and the shared
  `oversizedFileError` / `MAX_UPLOAD_MB` size validation
- Upload & Compare button → form action `/compare4/kml/upload`

The submit handler `startKmlUpload` validates selection + size, then performs a
native form submit. The server (`POST /compare4/kml/upload`) stores each track
with **no conversion step** — there is no progress UI because processing is
instant — and 303-redirects straight to the overlay map view. From there the
split view is one click away. KML comparisons are **map-only** (no graph data),
so the map views omit their `📊 Report` button for these groups.

Deep link: `?tab=kml` opens this tab directly.

### Tab 4: NMEA Comparison (`#tab-comparison`)
Upload two NMEA files (Reference + Test Device) for position error analysis.

**Form elements:**
- Slot labels: Reference NMEA (required) / Test Device NMEA (required)
- Drag-and-drop dropzones for each (`.nmea/.txt/.log`)
- Upload & Analyze button → `POST /analyze_nmea` (response HTML replaces current page)

### Tab 5: Reference vs DUT (`#tab-refdut`) — placeholder
Placeholder UI for comparing a Reference receiver log against a DUT `.ubx`
log. **Not implemented yet** — the reference log format is still undecided, so
no backend route exists.

- A "Work in progress" notice explains the pane is a placeholder.
- Slot labels: Reference Log (*format TBD*, accepts any file) / DUT UBX
  (required, `.ubx/.bin`). Both dropzones are wired via `wireDropzone` so the
  chosen filename displays, but nothing is uploaded.
- `Analyze (coming soon)` button is permanently `disabled`.

---

## JavaScript Structure

### Tab Switching (`openTab`)
```javascript
function openTab(tabName, btn) {
  // Hide all .tab-pane elements, remove active
  // Show target tab-pane
  // Add active to the clicked button
}
```

### URL Parameter Tab Activation
Checks query string on page load:
- `?tab=nmea` → activates NMEA tab
- `?tab=multi` → activates Multi tab
- (none) → defaults to Single tab

### Dropzone Wiring
- `wireDropzone(dropId, inputId, nameId)`: for Single/NMEA tabs
- `wireMultiDropzone(dropId, inputId, nameId)`: for Multi tab (also handles border color + opacity)
- Handles dragover / dragleave / drop events
- Injects dragged file into `input.files` via DataTransfer API
- **Client-side size check** — both wirers call `validatePickedFile(dz, fi, fn, file)` as soon as a file is dropped or selected. If `file.size > MAX_UPLOAD_BYTES` the input is cleared, the dropzone gets the `.oversized` class (red dashed border + red background), and the filename slot shows `"⚠ N.N MB — exceeds 1024 MB limit"`. The limit comes from `max_upload_mb` (injected by the `/` route from `MAX_UPLOAD_MB`, defaults to 1024).
- Each dropzone hint string includes the capacity (`"up to 1024 MB"`) so users see the limit before they pick a file.

### Single Upload (`startUpload`) & Progress UI
XHR upload (`uploadWithProgress`) followed by 1 s status polling
(`pollUntilDone`), driving the `prog-box` bar + step chips:

| Bar range | Phase | Source |
|---|---|---|
| 0 → 40% | 📤 Uploading… N% | XHR `upload.progress` events (real transfer progress). Files > `CHUNK_THRESHOLD_BYTES` (95 MB) are sent via `uploadInChunks()` — sequential 64 MB `POST /upload/chunk` requests + `POST /upload/complete` — to stay under Cloudflare's ~100 MB request-body cap on the tunnel; progress spans all chunks. Smaller files use the original single-request `/upload` |
| 45% | ⏳ Queued… | `/api/status` = `queued` |
| 50 → 95% | ⚙️ Processing… N% | `/api/status` = `running`; `progress` field = the converter's **real scan percentage** (read from the `.progress` sidecar `ubx2kmz --progress-file` writes ~every 0.5 s). Before the first sample the bar holds at 50% |
| 100% | ✅ Done | `done` → redirect to the report |

**Upload error surfacing** — `uploadWithProgress` resolves only for final
status < 400 (XHR follows the 303 redirect, so success lands on the report
page). For 4xx rejections the server's actual message body (rate-limit,
"Invalid file format: not a UBX binary", file-too-large) is thrown and shown
in the phase line — previously any 4xx surfaced as a generic
"Unexpected server response". `humanizeError()` reduces HTML error pages
(e.g. a Cloudflare 413) to their readable text before display.

The Multi tab reuses `pollUntilDone`; its per-file status rows append the same
real percentage while a file is converting ("Processing… 63%").

If the server restarts mid-conversion, the backend re-enqueues interrupted
jobs on startup (see `docs/app.md`), so the poll loop resumes/terminates
instead of spinning forever.

### NMEA Upload (`startComparison`)
POSTs via fetch, then writes the response HTML directly into the current page using `document.write()`.

---

## Design System (CSS)

Dark-blue theme:
- `--bg1: #0f1226`, `--bg2: #11163a`
- `--card: #151a44`
- `--accent: #6ea8ff`, `--accent2: #9d7bff`
- `--muted: #9bb2ff`, `--text: #e8eeff`

### Key Component Classes

| Class | Purpose |
|---|---|
| `.tabs` | Tab navigation row (flex, border-bottom) |
| `.tab-btn` | Tab button. `.active` applies accent background |
| `.tab-pane` | Tab content area. `.active` shows block + fadeIn animation |
| `.drop` | Drag-and-drop zone (dashed border, clickable label) |
| `.drop.dragover` | Highlighted border + scale(1.01) on drag-over |
| `.grid4` | 2-column CSS Grid for Multi tab's 4 dropzones |
| `.slot-label` | File slot title (uppercase, accent color) |
| `.required-tag` / `.optional-tag` | Required / optional indicator tags |
| `.notice` | Info box (dark background + border) |
| `.btn` | Action button (gradient blue) |
| `.spinner` | CSS loading ring animation |
| `.footer` | Bottom area for hints and links |
| `.chk` | Checkbox + label group |

---

## Dependencies (no external resources)
Pure HTML/CSS/JS. No CDN dependencies.
