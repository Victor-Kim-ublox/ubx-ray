# templates/compare4_overlay.html

## Overview
A map viewer that renders GPS tracks from up to 4 UBX files **overlaid on a single map**. Each track is distinguished by a unique color, and per-track visibility can be toggled individually.

---

## URL
`GET /compare4/overlay/{rid1}/{rid2}/{rid3}/{rid4}`

---

## Jinja2 Template Variables

| Variable | Type | Description |
|---|---|---|
| `rids` | list[str] | 4 result IDs (`"_"` for empty slots) |
| `filenames` | list[str] | Filename corresponding to each rid |

---

## Screen Layout

### Header (sticky, 60px)
- "4-Track Overlay" title
- Toolbar (flex, wrappable), grouped left-to-right like the single map view:
  1. **View controls** — `Map` / `Satellite` base toggle plus **Fit All** (combined in one segment; same grouping as the single map view).
  2. **Track toggles** — one colored badge button per file; click to hide/show that track's layer.
  3. **Playback** — `▶ Play` / `⏸ Pause`, speed selector (1× / 2× / 5× / 10×), timeline slider, UTC readout, `Follow`.
  4. **Distance measurement** — `Distance measure` (click two points for a Haversine read-out), `Clear` (remove the current line).
  5. **Right-side links** — `📊 Report` → `/compare4/report/{rids}`, `⊞ Split Map View` → `/compare4/view/{rids}`, `← New`.

### Map (`#map`)
Full-screen OpenLayers single map instance (viewport height minus header).

---

## Track Visibility Toggle (JavaScript)

On badge button click:
```javascript
layer.setVisible(!layer.getVisible());
btn.classList.toggle('dimmed');  // visual feedback for inactive state
```

---

## KML Loading and Layer Setup

For each active rid:
1. `fetch('/kml/{rid}')` → KML text
2. Parse with OpenLayers `new ol.format.KML()`
3. Create `new ol.layer.Vector({ source, style })`
4. Apply per-track color override (see below)

After all tracks are loaded, fit to the combined extent of all features:
```javascript
const combined = ol.extent.createEmpty();
layers.forEach(l => ol.extent.extend(combined, l.getSource().getExtent()));
map.getView().fit(combined, { padding: [60, 60, 60, 60] });
```

---

## Track Colors (KML Style Override)

KML's own styles are ignored; colors are forced by file index:

| File | Color | Hex |
|---|---|---|
| File 1 | blue | `#0078ff` |
| File 2 | red | `#e03e3e` |
| File 3 | green | `#16a34a` |
| File 4 | purple | `#9333ea` |

Point features are rendered as circular icons with uniform size and opacity.

---

## Difference vs. Split Map View

| Aspect | Split Map View | Overlay Map View |
|---|---|---|
| Map instances | 4 (one per panel) | 1 (single instance) |
| Track display | Each track in its own panel | All 4 tracks on one map |
| Sync required | Yes (shared view) | No (single instance) |
| Use case | Detailed inspection of individual tracks | Route comparison / overlap analysis |

---

## Dependencies (CDN)
```html
<link href="https://cdn.jsdelivr.net/npm/ol@9.1.0/ol.css">
<script src="https://cdn.jsdelivr.net/npm/ol@9.1.0/dist/ol.js">
```
Google Fonts Inter.

---

## Design
- Light theme (`--bg: #f8fafc`)
- White header with drop shadow
- Track toggle buttons: per-file color badge style inside `.seg` group
- Inactive tracks: `.dimmed` class (opacity 0.4, line-through text)
