# templates/map.html

## Overview
A **KMZ-based map viewer** for single UBX file analysis results. Renders KML points via OpenLayers with support for satellite/road map switching, point size adjustment, and timeline playback.

---

## URL
`GET /map/{rid}`

---

## Jinja2 Template Variables

| Variable | Description |
|---|---|
| `rid` | Result ID (8-character hex) |
| `filename` | Original UBX filename (shown in page title) |

---

## Screen Layout

### Header (sticky, 60px)
- "Map View" title + filename
- **Toolbar** (flex) — grouped left-to-right by function:
  1. **View controls** — `Map` / `Satellite` base map toggle, and **Fit** which re-frames the map to the current track extent (padding 20 px, max zoom 17, 250 ms animation). Fit shares its `fitToTrack()` implementation with the initial auto-fit that runs when the KML finishes loading.
  2. **Playback** — `▶ Play` / `⏸ Pause`, speed selector (1× / 2× / 5× / 10×), timeline slider, UTC time readout, `Follow` (keep marker centered).
  3. **Distance measurement** — `Distance measure` (click two points for a Haversine read-out), `Clear` (remove the current line).
  4. **Jam/Spoof overlay** (shown only when SEC-SIG data exists) — toggle button plus inline legend for `Jam`, `Spf ind.`, `Spf aff.`
  5. **Links** — `📋 Report` → `/report/{rid}`, `⬇ Download KMZ`.

### Map (`#map`)
Full-screen (viewport height minus header height).

---

## Map Layer Structure

### Base Layers
| Name | Source | Description |
|---|---|---|
| Satellite | Bing Maps Aerial | Satellite imagery (default) |
| Road | OSM | OpenStreetMap road map |

Switching is done by clicking `.seg` buttons → only the selected layer is set to `visible(true)`.

### Track Layer (Vector Layer)
Fetches KML data from `/kml/{rid}` → parses with OpenLayers `KML` format.

Each Placemark in the KML:
- `<TimeStamp>` → time information (used for playback)
- `<Style>/<IconStyle>/<color>` → fixType-based color (green/yellow/red)
- `<description>` → popup tooltip content

---

## Interaction Features

### Point Size Adjustment
On slider value change, re-applies style to all features in the Vector Layer:
```javascript
vectorSource.getFeatures().forEach(feature => {
  const style = feature.getStyle() || defaultStyle;
  style.getImage().setScale(sliderValue);
  feature.setStyle(style);
});
```

### Timeline Playback
Sequentially reveals points based on their TimeStamp:
```javascript
// Sort points by time
// setInterval to show next point every N ms
// Pan map to current point
// Adjust interval based on selected playback speed
```

Current point metadata (UTC time, coordinates, speed, etc.) is displayed on screen during playback.

### Feature Click Tooltip
Clicking a point shows the KML `<description>` content in a popup:
- UTC, iTOW, FixType, Flags
- Heading, HeadAcc
- Speed (m/s, km/h), SpeedAcc
- Lat, Lon, PosAcc2D
- Alt, AltAcc

---

### Jamming / Spoofing Overlay (UBX-SEC-SIG)

When the graph JSON at `/api/graph/{rid}` contains non-empty `sec_labels`, an additional vector layer is added on top of the track. Each SEC-SIG sample is matched back to a PVT epoch via `iTOW`, and a marker is drawn at that epoch's `lat/lon`:

Rendered as **track highlights** rather than per-epoch point markers. For each of the three conditions below, contiguous epochs are grouped into runs and drawn as LineStrings along the vehicle track. An iTOW gap larger than 3000 ms closes the current run so missing data is not bridged by a straight line. An isolated single-epoch run falls back to a small dot so it stays visible.

| Condition | Track overlay |
|---|---|
| `jam_state == 2` (Warning)  | Red zone sheen: wide translucent red band (rgba(230,0,0,0.28), 26 px, solid, round caps) drawn **on top** of the track so the arrows show through as a red-tinted hazard zone. Pixel-constant width keeps the zone readable at every zoom level |
| `spf_state == 2` (Indicated) | Dashed chartreuse line (#CCFF00, 3.5 px, over the track arrows) |
| `spf_state == 3` (Affirmed)  | Dashed cyan line (#00E5FF, 4 px, dash `[14, 5]`, over the track arrows) |

Layer stack (bottom → top), by OpenLayers `zIndex`:

| zIndex | Layer | Purpose |
|---|---|---|
| 410 | vectorLayer | Track arrows — base layer for SEC-SIG overlays |
| 415 | jamLayer | Jamming red zone sheen (wide translucent red) — sits above the arrows to tint the segment as a hazard zone, but below the spoofing lines so those stay readable on top |
| 420 | spfIndLayer | Spoofing indicated (dashed chartreuse) — thin line on top of everything below |
| 440 | spfAffLayer | Spoofing affirmed (dashed cyan) — thin line on top |
| 600 | measureLayer | Distance-measure line |
| 700 | markerLayer | Current playback marker |

The three SEC-SIG overlays live on separate vector layers with distinct z-indices (jam halo → spfIndicated → spfAffirmed). Concurrent jamming + spoofing on the same segment therefore shows up as a red halo with an orange or magenta line running along its center — both conditions remain readable. A "Jam/Spoof" toolbar segment is rendered only when at least one run is produced; it contains a toggle button (hides/shows all three layers together) and an inline SVG legend. Clicking a segment opens the popup with the run's iTOW range and epoch count.

---

## KMZ Download
"Download KMZ" button → `GET /download?path={kmz_path}` (server verifies ownership before returning the file).

---

## Dependencies (CDN)
```html
<link href="https://cdn.jsdelivr.net/npm/ol@latest/ol.css">
<script src="https://cdn.jsdelivr.net/npm/ol@latest/dist/ol.js">
```
Google Fonts Inter.

> **Note**: Uses `ol@latest`. `compare4_view.html` and `compare4_overlay.html` use pinned version `ol@9.1.0`.

---

## Design
Light theme:
- `--bg: #f8fafc`, `--border: #e5e7eb`, `--ink: #1e293b`
- `--accent: #0078ff`
- White header with drop shadow
