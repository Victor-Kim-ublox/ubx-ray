# templates/map.html

## Overview
A **KMZ-based map viewer** for single UBX file analysis results. Renders KML points via OpenLayers with support for satellite/road map switching, point size adjustment, and a
two-handle Range selector that trims which part of the track is drawn.

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
  2. **Range** — a two-handle slider that picks the slice of the track to draw (see Track Range below), a UTC read-out of the selected span, and `Full` to restore the whole path.
  3. **Distance measurement** — `Distance measure` (click two points for a Haversine read-out), `Clear` (removes the current line **and** the Distance result popup; the shared `#popup` is only hidden when it is showing a Distance result, so an open SEC-SIG popup is left untouched).
  4. **Jam/Spoof overlay** (shown only when SEC-SIG data exists) — toggle button plus inline legend for `Jam`, `Spf ind.`, `Spf aff.`
  5. **Links** — `📋 Report` → `/report/{rid}`, `⬇ Download KMZ`.

### Map (`#map`)
Full-screen (viewport height minus header height).

### Loading overlay (`#loadingOverlay`)
A fixed full-screen overlay (blurred backdrop + spinner + "Loading map…" /
"Fetching track data" + a slim progress bar) shown on first paint while the
KML track is fetched. The download streams through a `ReadableStream` reader
and reports a **real percentage** ("Downloading track… 43%"): gzip makes
`Content-Length` the compressed size while fetch yields decompressed bytes,
so `/kml/{rid}` sends the raw size in an `X-Uncompressed-Size` header and the
bar tracks decompressed-received / raw-size. The bar then resets for a second
stage, "Parsing track… N%": the generated KML is a flat list of
self-contained `<Placemark>` blocks, so it is split with an `indexOf` scan and
parsed in 4,000-placemark batches (each wrapped in a minimal KML envelope),
updating the bar and yielding to the event loop between batches — a single
`readFeatures()` call on a 100+ MB document used to block the main thread for
many seconds with the bar stuck at 100 %. Non-flat KMLs (e.g. gx:Track)
fall back to the single-call parse, and files under one batch skip the
splitting entirely.
Matters now that uploads can be up to 1 GB — the derived KMZ can take a
moment to load. `hideLoading()` removes it when the track layer's
`vectorSource` reaches the `ready` state (the real "track on screen" moment),
on its `error` state, and as a fallback when `loadData()` resolves/rejects, so
the user is never left behind a stuck spinner.

---

## Map Layer Structure

### Base Layers
| Name | Source | Description |
|---|---|---|
| Satellite | Bing Maps Aerial | Satellite imagery (default) |
| Road | OSM | OpenStreetMap road map |

Switching is done by clicking `.seg` buttons → only the selected layer is set to `visible(true)`.

### Track Layer (Vector Layer)
`loadData()` fetches `/kml/{rid}` **once** and parses it **once** with the
OpenLayers `KML` format (the source previously also downloaded/parsed the
same KML through its own url-loader — a 100+ MB XML for large logs, so the
double transfer/parse dominated load time). The server gzips the response
(~30× smaller). Per-point timestamps for the Range read-out are extracted with
a linear `<when>` regex scan paired to the parsed point features
(`buildTrackTimes()`) — no DOM tree is built for the whole document. When the
`<when>` count does not match the track features the array is left empty and
the Range label falls back to point counts.

**Zoom-adaptive rendering** — all parsed features stay in memory, but the
vector source only holds what is worth drawing for the current view: at most
`MAX_RENDER = 6000` track arrows inside the (10 %-padded) viewport, picked
with an even stride and refreshed on `moveend`. Zooming in shrinks the
in-view set, so detail increases until every point in view renders. AID-MAPM
markers and non-point geometries always render; the last in-view point is
always kept so the track end never disappears. The Range selection narrows the
eligible slice **before** this viewport filter runs, so trimming and
zoom-adaptive thinning compose. Popups work on the rendered subset.

Each Placemark in the KML:
- `<TimeStamp>` → time information (used for the Range read-out)
- `<Style>/<IconStyle>/<color>` → fixType-based color (green/yellow/red)
- `<description>` → popup tooltip content

**AID-MAPM markers** — the KML may also contain sky-blue
(`FFEBCE87`) `<name>AID-MAPM</name>` arrow Placemarks (map-matching points
parsed from UBX-AID-MAPM). They render on the same vector layer and open the
same click popups. Every AID-MAPM point shows the iTOW of the NAV-PVT that
preceded it in the stream as `arrived iTOW` (right under the message `iTOW`) so
it is clear after which NAV-PVT the message arrived. For `relativePos` points
the delta is anchored to the NAV-PVT fix at the *same iTOW* as the message, and
the popup additionally shows the raw delta (`ΔLat`/`ΔLon`) and the resulting
`computed` absolute Lat/Lon; absolute points show a plain Lat/Lon. AID-MAPM
Placemarks are kept out of the indexed track array (they carry no
`<TimeStamp>` and are markers rather than vehicle-track epochs), so they are
**not** affected by the Range trim and always render.

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

### Track Range (trim which part of the path is drawn)
Replaces the old timeline playback (play/pause, speed, follow marker), which
went unused. Two `<input type="range">` elements are stacked transparently over
one painted rail (`.range-wrap`); only their thumbs take pointer events, so each
handle drags independently. The values are **inclusive indices into
`trackFeatures`** (document order), so the full span is the whole path:

- Drag the **left** handle to cut the beginning, the **right** handle to cut the
  end. `onRangeInput()` stops a handle at the other one rather than letting them
  cross.
- The **selected handle is highlighted** (filled in the accent colour with a
  soft ring) and grows slightly while it is being dragged, so it is always clear
  which end is moving. "Selected" means the focused handle, falling back to the
  one last dragged: the highlight is driven by a `.sel` class
  (`updateHandleStacking()`) as well as `:focus`, because a range input can be
  driven without the window itself holding focus. `updateHandleStacking()` also
  raises the selected handle above the other so its ring is never clipped;
  with neither selected, the handle in the right half stays on top so a pair
  that lands on the same spot can still be pulled apart.
- The read-out shows the selected UTC span (`12:50:50Z → 12:53:50Z`) when
  `trackTimes` is available, otherwise `shown / total pts`; its tooltip always
  carries both the counts and the point indices.
- **`Full`** restores the whole track.

`applyRange()` updates the label immediately (so dragging feels live) and
coalesces the heavier redraw to one per frame — a drag fires many `input`
events and each redraw walks the selected slice. In a hidden tab
`requestAnimationFrame` never fires, so the redraw runs straight away instead.

What the selection affects:

| Area | Behaviour |
|---|---|
| Track arrows | Only indices in `[selLo(), selHi()]` are eligible; the zoom-adaptive viewport filter then thins that slice |
| `Fit` / initial fit | `shownExtent()` frames the **selected** slice, not the whole log |
| Jam/spoof overlays | `rebuildSecOverlays()` re-cuts every run to the selected iTOW window, so the highlights stop exactly where the drawn track stops (see below) |
| Popups, AID-MAPM markers | Unaffected — AID-MAPM is not part of the indexed track |

### Feature Click Popups (multiple, stacked)
Clicking a point spawns an **independent popup** anchored to that point's
marker, showing the KML `<description>` content:
- UTC, iTOW, FixType, Flags
- Heading, HeadAcc
- Speed (m/s, km/h), SpeedAcc
- Lat, Lon, PosAcc2D
- Alt, AltAcc

Several popups can stay open at once (for comparison) — each click **adds** one
rather than replacing the previous. Each popup closes via its own `×`; a
**Close popups** toolbar button (shown only while any are open) clears them all.

Each popup is tied to its point by a **numbered ring marker** (coloured, on
`pinLayer` zIndex 720) dropped on the clicked point; the popup shows the **same
coloured number badge** (`.pt-num`) plus a matching top border — so it is
always clear which popup belongs to which point. Numbers are the smallest free
integer (reused on close); colours cycle through `POPUP_COLORS`.

New popups appear at a fixed offset just above their point (they may overlap if
points are close). Each popup is **draggable by its title bar**
(`makeDraggable()`, pointer events with pointer capture) — dragging adjusts the
overlay's pixel offset, so the user repositions overlapping popups by hand while
each still tracks its point on zoom/pan. `stopEvent` keeps the map from panning
during a drag.

`addPointPopup()` creates a fresh `ol.Overlay` per click (`stopEvent:true`,
`autoPan:false`); descriptions run through `sanitizeHtml()` first. Closing a
popup removes both its overlay and its ring marker.

SEC-SIG (jam/spoof) segment clicks spawn the **same stacked, draggable
popups** as track points (`addPointPopup()`), so several jam/spoof segments
can be compared side by side instead of each click replacing the previous.
They pass `{ badge: false }`, so — unlike the track-point popups — they carry
**no number badge**, their ring marker is an unnumbered coloured ring (the
colour still ties each popup border to its marker), and they **do not consume
a number**: track-point popups keep counting 1, 2, 3… regardless of how many
jam/spoof popups are open. Only the
distance-measurement result still uses the single shared `#popup` overlay,
which is **also draggable** by its title (same `makeDraggable`); its offset
resets to the default each time it is shown.

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

The three SEC-SIG overlays live on separate vector layers with distinct z-indices (jam halo → spfIndicated → spfAffirmed). Concurrent jamming + spoofing on the same segment therefore shows up as a red halo with an orange or magenta line running along its center — both conditions remain readable. A "Jam/Spoof" toolbar segment is rendered only when at least one run is produced; it contains a toggle button (hides/shows all three layers together) and an inline SVG legend. Clicking a segment opens a stacked popup (via `addPointPopup` with `{ badge: false }` — no number) with the run's iTOW range and epoch count; multiple segment popups can stay open at once.

**Range clipping** — the parsed runs are kept in `secRuns` (each carrying its
per-point `itows[]` alongside `coords[]`), and `rebuildSecOverlays()` redraws
the three layers whenever the Range selection changes, keeping only the points
whose iTOW falls inside the selected window. The clip is therefore exact: a jam
zone that extends past the trimmed track is cut at the same place, and runs
entirely outside the selection disappear. Track index → iTOW comes from the
graph JSON's `labels`, which lines up index-for-index with the KML track
placemarks (both are emitted in the same conversion pass); if the counts ever
disagree the clip is skipped and runs are drawn whole. The Jam/Spoof toolbar
segment stays visible based on the **total** run count, so trimming to a clean
stretch never hides the toggle.

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
