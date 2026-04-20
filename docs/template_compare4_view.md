# templates/compare4_view.html

## Overview
A map viewer that displays GPS tracks from up to 4 UBX files in a **2×2 split-screen layout (Split Map View)**. Built on OpenLayers, with automatic zoom/position synchronization across all panels.

---

## URL
`GET /compare4/view/{rid1}/{rid2}/{rid3}/{rid4}`

---

## Jinja2 Template Variables

| Variable | Type | Description |
|---|---|---|
| `rids` | list[str] | 4 result IDs (`"_"` for empty slots) |
| `filenames` | list[str] | Filename corresponding to each rid |

---

## Screen Layout

### Header (sticky, 54px)
- ubX-ray brand
- **View controls** segment (grouped like the single map view): `Map` / `Satellite` base toggle + `Fit All` (fits every panel to its track extent).
- **Jam/Spoof segment** (`#secSeg`, hidden until at least one panel produces a SEC-SIG run). Contains a single toggle button that hides/shows jam + spoofing overlays across **all four panels at once**, plus the same SVG legend used by the single map view (Jam / Spf ind. / Spf aff.).
- Right-side links: `📊 Report`, `⊙ Overlay Map View` → `/compare4/overlay/{rids}`.

### Map Grid (`#grid`)
`grid-template-columns: 1fr 1fr` — 2 columns × 2 rows = up to 4 panels.

Each panel:
- Filename overlay badge (top-left, per-file color)
- Empty slots (`"_"`) show a dark background with "Empty slot" text only

---

## Map Synchronization Logic (JavaScript)

### KML Loading
Fetches KML data from `/kml/{rid}` for each rid, parses it with OpenLayers `KML` format, and adds it as a Vector Layer.

```javascript
// Fit map view to the loaded track extent after KML is loaded
map.getView().fit(vectorSource.getExtent(), { padding: [40, 40, 40, 40] });
```

### View Synchronization
4 Map instances share center/zoom via **master-slave** sync:
```javascript
// On view change in map[0]
maps.forEach((m, i) => {
  if (i !== sourceIdx) {
    m.getView().setCenter(center);
    m.getView().setZoom(zoom);
    m.getView().setRotation(rotation);
  }
});
```
A `syncing` flag prevents recursive re-triggering.

---

## SEC-SIG (Jamming / Spoofing) Overlays

Each panel owns its own trio of vector layers (`jamLayer`, `spfIndLayer`, `spfAffLayer`) with the same styling and z-index ordering as `templates/map.html`:

| zIndex | Layer | Style |
|---|---|---|
| 415 | jamLayer (per panel) | Red zone sheen (rgba(230,0,0,0.28), 26 px) — on top of the track arrows |
| 420 | spfIndLayer (per panel) | Dashed chartreuse `#CCFF00` (3.5 px, `[8, 5]`) |
| 440 | spfAffLayer (per panel) | Dashed cyan `#00E5FF` (4 px, `[14, 5]`) |

After a panel's KML finishes loading, `loadSecSigPanel(idx, rid)` fetches `/api/graph/{rid}`, maps each SEC-SIG sample back to a PVT `(lat, lon)` via `iTOW`, then builds contiguous runs for each condition. An iTOW gap larger than 3000 ms breaks the current run; isolated single-epoch runs fall back to a dot feature so they stay visible.

Clicking a SEC-SIG feature opens that panel's existing popup with the run's iTOW range, epoch count, and a panel-labelled title (e.g. "Jamming Warning — Panel 2"). The header's Jam/Spoof toggle fires `setVisible()` on all 12 layers (3 × 4 panels) simultaneously.

---

## KML Styling

Beyond the default OpenLayers KML styles, track points are color-differentiated by file index:
- File 1: blue (`#0078ff`)
- File 2: red (`#e03e3e`)
- File 3: green (`#16a34a`)
- File 4: purple (`#9333ea`)

---

## Status Handling

For files whose conversion is not yet complete (KMZ not available):
- Panel shows "Converting... please wait"
- Retry via auto-refresh or polling (depending on implementation)

---

## Dependencies (CDN)
```html
<link href="https://cdn.jsdelivr.net/npm/ol@9.1.0/ol.css">
<script src="https://cdn.jsdelivr.net/npm/ol@9.1.0/dist/ol.js">
```
Google Fonts Inter.

---

## Design Theme
Light theme (shared with report and overlay pages):
- `--bg: #f8fafc`, `--border: #e5e7eb`, `--ink: #1e293b`
- Header: white background with drop shadow
- Panel background: map tile layer
