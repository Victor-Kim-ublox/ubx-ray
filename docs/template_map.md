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
- **Toolbar** (flex):
  - Map type segment: `Satellite` / `Road`
  - Point size slider (range input, 0.1–2.0)
  - Playback controls: `▶ Play` / `⏸ Pause` / `⏹ Reset`
  - Playback speed selector (1x / 2x / 5x / 10x)
  - Buttons: `📋 Report` → `/report/{rid}`, `⬇ Download KMZ`

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
