# templates/home.html

## Overview
The **main home screen** of ubX-ray. Provides three analysis tabs (Single / Multi / NMEA) in a single-page SPA-style interface.

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

### Tab 3: NMEA Comparison (`#tab-comparison`)
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
