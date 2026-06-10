# templates/home.html

## Overview
The **main home screen** of ubX-ray. Provides four analysis tabs
(Single / Multi / KML Comparison / NMEA) in a single-page SPA-style interface.
Only Single, Multi, and KML Comparison have tab buttons in the nav; the NMEA
pane is reachable via `?tab=nmea`.

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
- **Client-side size check** — both wirers call `validatePickedFile(dz, fi, fn, file)` as soon as a file is dropped or selected. If `file.size > MAX_UPLOAD_BYTES` the input is cleared, the dropzone gets the `.oversized` class (red dashed border + red background), and the filename slot shows `"⚠ N.N MB — exceeds 300 MB limit"`. The limit comes from `max_upload_mb` (injected by the `/` route from `MAX_UPLOAD_MB`, defaults to 300).
- Each dropzone hint string includes the capacity (`"up to 300 MB"`) so users see the limit before they pick a file.

### Single Upload (`startUpload`)
Async upload via fetch API. On response, redirects to `redirect_url` if present; otherwise reloads. Disables button and shows spinner during upload.

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
